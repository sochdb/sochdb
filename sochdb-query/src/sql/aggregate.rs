// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! # SQL Aggregation Executor
//!
//! Hash-aggregation operator for `GROUP BY` and aggregate functions.
//!
//! Supported aggregates: `COUNT(*)`, `COUNT(col)`, `COUNT(DISTINCT col)`,
//! `SUM`, `AVG`, `MIN`, `MAX`, `MEDIAN`, `STDDEV` (sample, n-1, matching
//! R's `sd()` and DuckDB's `stddev`).
//!
//! ## Pipeline
//!
//! ```text
//! input rows (post-WHERE)
//!   └─> group keys evaluated per row ──> hash table of group states
//!         └─> accumulators updated per row
//!               └─> finalize: one synthesized row per group
//!                     └─> HAVING filter ─> ORDER BY ─> OFFSET/LIMIT ─> projection
//! ```
//!
//! Semantics notes:
//! - NULL inputs are skipped by all aggregates except `COUNT(*)` (SQL standard).
//! - An ungrouped aggregate over zero rows yields exactly one row
//!   (`COUNT` = 0, other aggregates NULL); a grouped aggregate over zero
//!   rows yields zero rows.
//! - Non-aggregate SELECT columns that are not in `GROUP BY` resolve to the
//!   first value seen in the group (lenient mode, like SQLite / MySQL with
//!   `ONLY_FULL_GROUP_BY` disabled).

use super::ast::*;
use super::bridge::ExecutionResult;
use super::error::SqlResult;
use rayon::prelude::*;
use sochdb_core::SochValue;
use std::collections::{HashMap, HashSet};

/// Row count above which grouped accumulation runs on the rayon pool.
const PARALLEL_THRESHOLD: usize = 100_000;

// ============================================================================
// Aggregate function identification
// ============================================================================

/// Recognized aggregate functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFn {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Median,
    Stddev,
}

impl AggFn {
    /// Recognize an aggregate function by name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_uppercase().as_str() {
            "COUNT" => Some(Self::Count),
            "SUM" => Some(Self::Sum),
            "AVG" | "MEAN" => Some(Self::Avg),
            "MIN" => Some(Self::Min),
            "MAX" => Some(Self::Max),
            "MEDIAN" => Some(Self::Median),
            "STDDEV" | "STDDEV_SAMP" | "STDEV" | "SD" => Some(Self::Stddev),
            _ => None,
        }
    }
}

/// One aggregate call discovered in the query, e.g. `sum(v1)`.
#[derive(Debug, Clone)]
struct AggSpec {
    /// Canonical key, e.g. `"sum(v1)"` — used to bind HAVING / ORDER BY
    /// references back to the computed value.
    key: String,
    func: AggFn,
    /// Argument expression (`None` for `COUNT(*)`).
    arg: Option<Expr>,
    distinct: bool,
}

/// Returns true if the SELECT needs the aggregation operator.
pub fn is_aggregate_query(select: &SelectStmt) -> bool {
    if !select.group_by.is_empty() {
        return true;
    }
    select
        .columns
        .iter()
        .any(|item| matches!(item, SelectItem::Expr { expr, .. } if contains_aggregate(expr)))
}

/// Recursively check whether an expression contains an aggregate call.
fn contains_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Function(f) => {
            AggFn::from_name(f.name.name()).is_some()
                || f.args.iter().any(contains_aggregate)
        }
        Expr::BinaryOp { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        Expr::UnaryOp { expr, .. } => contains_aggregate(expr),
        Expr::IsNull { expr, .. } => contains_aggregate(expr),
        Expr::Case {
            operand,
            conditions,
            else_result,
        } => {
            operand.as_deref().map(contains_aggregate).unwrap_or(false)
                || conditions
                    .iter()
                    .any(|(w, t)| contains_aggregate(w) || contains_aggregate(t))
                || else_result
                    .as_deref()
                    .map(contains_aggregate)
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// Collect all distinct aggregate calls from SELECT, HAVING and ORDER BY.
fn collect_agg_specs(select: &SelectStmt) -> Vec<AggSpec> {
    let mut specs: Vec<AggSpec> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let walk = |expr: &Expr, specs: &mut Vec<AggSpec>, seen: &mut HashSet<String>| {
        collect_from_expr(expr, specs, seen);
    };

    for item in &select.columns {
        if let SelectItem::Expr { expr, .. } = item {
            walk(expr, &mut specs, &mut seen);
        }
    }
    if let Some(h) = &select.having {
        walk(h, &mut specs, &mut seen);
    }
    for ob in &select.order_by {
        walk(&ob.expr, &mut specs, &mut seen);
    }
    specs
}

fn collect_from_expr(expr: &Expr, specs: &mut Vec<AggSpec>, seen: &mut HashSet<String>) {
    match expr {
        Expr::Function(f) => {
            if let Some(func) = AggFn::from_name(f.name.name()) {
                let arg = f.args.first().cloned();
                let is_star =
                    matches!(arg.as_ref(), Some(Expr::Column(c)) if c.column == "*");
                let arg = if is_star { None } else { arg };
                let key = render_agg_key(func, arg.as_ref(), f.distinct);
                if seen.insert(key.clone()) {
                    specs.push(AggSpec {
                        key,
                        func,
                        arg,
                        distinct: f.distinct,
                    });
                }
            } else {
                for a in &f.args {
                    collect_from_expr(a, specs, seen);
                }
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_from_expr(left, specs, seen);
            collect_from_expr(right, specs, seen);
        }
        Expr::UnaryOp { expr, .. } => collect_from_expr(expr, specs, seen),
        Expr::IsNull { expr, .. } => collect_from_expr(expr, specs, seen),
        Expr::Case {
            operand,
            conditions,
            else_result,
        } => {
            if let Some(op) = operand {
                collect_from_expr(op, specs, seen);
            }
            for (w, t) in conditions {
                collect_from_expr(w, specs, seen);
                collect_from_expr(t, specs, seen);
            }
            if let Some(e) = else_result {
                collect_from_expr(e, specs, seen);
            }
        }
        _ => {}
    }
}

/// Canonical name for an aggregate call: `sum(v1)`, `count(*)`,
/// `count(distinct id)`. Lowercased so lookups are case-insensitive.
fn render_agg_key(func: AggFn, arg: Option<&Expr>, distinct: bool) -> String {
    let fname = match func {
        AggFn::Count => "count",
        AggFn::Sum => "sum",
        AggFn::Avg => "avg",
        AggFn::Min => "min",
        AggFn::Max => "max",
        AggFn::Median => "median",
        AggFn::Stddev => "stddev",
    };
    let arg_s = match arg {
        None => "*".to_string(),
        Some(e) => render_expr_name(e),
    };
    if distinct {
        format!("{}(distinct {})", fname, arg_s)
    } else {
        format!("{}({})", fname, arg_s)
    }
}

/// Human-readable name for an expression, used for output column naming
/// and canonical aggregate keys.
pub fn render_expr_name(expr: &Expr) -> String {
    match expr {
        Expr::Column(c) => {
            if let Some(t) = &c.table {
                format!("{}.{}", t, c.column)
            } else {
                c.column.clone()
            }
        }
        Expr::Literal(Literal::Integer(n)) => n.to_string(),
        Expr::Literal(Literal::Float(f)) => f.to_string(),
        Expr::Literal(Literal::String(s)) => format!("'{}'", s),
        Expr::Literal(Literal::Boolean(b)) => b.to_string(),
        Expr::Literal(Literal::Null) => "null".to_string(),
        Expr::Function(f) => {
            if let Some(func) = AggFn::from_name(f.name.name()) {
                let arg = f.args.first();
                let is_star =
                    matches!(arg, Some(Expr::Column(c)) if c.column == "*");
                render_agg_key(func, if is_star { None } else { arg }, f.distinct)
            } else {
                let args: Vec<String> = f.args.iter().map(render_expr_name).collect();
                format!("{}({})", f.name.name().to_lowercase(), args.join(", "))
            }
        }
        Expr::BinaryOp { left, op, right } => format!(
            "{} {} {}",
            render_expr_name(left),
            binary_op_symbol(op),
            render_expr_name(right)
        ),
        Expr::UnaryOp { op, expr } => match op {
            UnaryOperator::Minus => format!("-{}", render_expr_name(expr)),
            UnaryOperator::Plus => render_expr_name(expr),
            UnaryOperator::Not => format!("not {}", render_expr_name(expr)),
            UnaryOperator::BitNot => format!("~{}", render_expr_name(expr)),
        },
        _ => "expr".to_string(),
    }
}

fn binary_op_symbol(op: &BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::Plus => "+",
        BinaryOperator::Minus => "-",
        BinaryOperator::Multiply => "*",
        BinaryOperator::Divide => "/",
        BinaryOperator::Modulo => "%",
        BinaryOperator::Eq => "=",
        BinaryOperator::Ne => "<>",
        BinaryOperator::Lt => "<",
        BinaryOperator::Le => "<=",
        BinaryOperator::Gt => ">",
        BinaryOperator::Ge => ">=",
        BinaryOperator::And => "and",
        BinaryOperator::Or => "or",
        _ => "?",
    }
}

// ============================================================================
// Scalar expression evaluation (over a materialized row)
// ============================================================================

/// Evaluate a scalar expression against a row map.
///
/// `agg_values`, when provided, resolves aggregate function calls by their
/// canonical key — used for HAVING / ORDER BY / projection over finalized
/// group rows.
fn eval_scalar(
    expr: &Expr,
    row: &HashMap<String, SochValue>,
    params: &[SochValue],
) -> SochValue {
    match expr {
        Expr::Column(c) => {
            if let Some(t) = &c.table {
                let qualified = format!("{}.{}", t, c.column);
                if let Some(v) = row.get(&qualified) {
                    return v.clone();
                }
            }
            row.get(&c.column).cloned().unwrap_or(SochValue::Null)
        }
        Expr::Literal(lit) => literal_to_value(lit),
        Expr::Placeholder(idx) => params
            .get((*idx as usize).saturating_sub(1))
            .cloned()
            .unwrap_or(SochValue::Null),
        Expr::Function(f) => {
            // Aggregate results are pre-bound into the row map under their
            // canonical key by `finalize_groups`.
            let key = render_expr_name(&Expr::Function(f.clone()));
            row.get(&key).cloned().unwrap_or(SochValue::Null)
        }
        Expr::BinaryOp { left, op, right } => {
            let l = eval_scalar(left, row, params);
            let r = eval_scalar(right, row, params);
            eval_binary(&l, op, &r)
        }
        Expr::UnaryOp { op, expr } => {
            let v = eval_scalar(expr, row, params);
            match op {
                UnaryOperator::Minus => match v {
                    SochValue::Int(i) => SochValue::Int(-i),
                    SochValue::Float(f) => SochValue::Float(-f),
                    _ => SochValue::Null,
                },
                UnaryOperator::Plus => v,
                UnaryOperator::Not => match v {
                    SochValue::Bool(b) => SochValue::Bool(!b),
                    _ => SochValue::Null,
                },
                UnaryOperator::BitNot => match v {
                    SochValue::Int(i) => SochValue::Int(!i),
                    _ => SochValue::Null,
                },
            }
        }
        Expr::IsNull { expr, negated } => {
            let v = eval_scalar(expr, row, params);
            let is_null = v.is_null();
            SochValue::Bool(if *negated { !is_null } else { is_null })
        }
        _ => SochValue::Null,
    }
}

fn literal_to_value(lit: &Literal) -> SochValue {
    match lit {
        Literal::Integer(i) => SochValue::Int(*i),
        Literal::Float(f) => SochValue::Float(*f),
        Literal::String(s) => SochValue::Text(s.clone()),
        Literal::Boolean(b) => SochValue::Bool(*b),
        Literal::Null => SochValue::Null,
        _ => SochValue::Null,
    }
}

fn numeric(v: &SochValue) -> Option<f64> {
    match v {
        SochValue::Int(i) => Some(*i as f64),
        SochValue::UInt(u) => Some(*u as f64),
        SochValue::Float(f) => Some(*f),
        SochValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn eval_binary(l: &SochValue, op: &BinaryOperator, r: &SochValue) -> SochValue {
    use BinaryOperator::*;
    match op {
        Plus | Minus | Multiply | Divide | Modulo => {
            // Integer arithmetic when both sides are ints (except division).
            if let (SochValue::Int(a), SochValue::Int(b)) = (l, r) {
                return match op {
                    Plus => SochValue::Int(a.wrapping_add(*b)),
                    Minus => SochValue::Int(a.wrapping_sub(*b)),
                    Multiply => SochValue::Int(a.wrapping_mul(*b)),
                    Divide => {
                        if *b == 0 {
                            SochValue::Null
                        } else {
                            SochValue::Float(*a as f64 / *b as f64)
                        }
                    }
                    Modulo => {
                        if *b == 0 {
                            SochValue::Null
                        } else {
                            SochValue::Int(a % b)
                        }
                    }
                    _ => unreachable!(),
                };
            }
            let (a, b) = match (numeric(l), numeric(r)) {
                (Some(a), Some(b)) => (a, b),
                _ => return SochValue::Null,
            };
            match op {
                Plus => SochValue::Float(a + b),
                Minus => SochValue::Float(a - b),
                Multiply => SochValue::Float(a * b),
                Divide => {
                    if b == 0.0 {
                        SochValue::Null
                    } else {
                        SochValue::Float(a / b)
                    }
                }
                Modulo => {
                    if b == 0.0 {
                        SochValue::Null
                    } else {
                        SochValue::Float(a % b)
                    }
                }
                _ => unreachable!(),
            }
        }
        Eq | Ne | Lt | Le | Gt | Ge => {
            if l.is_null() || r.is_null() {
                return SochValue::Null;
            }
            let ord = compare_values(l, r);
            let b = match op {
                Eq => ord == std::cmp::Ordering::Equal,
                Ne => ord != std::cmp::Ordering::Equal,
                Lt => ord == std::cmp::Ordering::Less,
                Le => ord != std::cmp::Ordering::Greater,
                Gt => ord == std::cmp::Ordering::Greater,
                Ge => ord != std::cmp::Ordering::Less,
                _ => unreachable!(),
            };
            SochValue::Bool(b)
        }
        And => match (as_bool(l), as_bool(r)) {
            (Some(a), Some(b)) => SochValue::Bool(a && b),
            _ => SochValue::Null,
        },
        Or => match (as_bool(l), as_bool(r)) {
            (Some(a), Some(b)) => SochValue::Bool(a || b),
            _ => SochValue::Null,
        },
        _ => SochValue::Null,
    }
}

fn as_bool(v: &SochValue) -> Option<bool> {
    match v {
        SochValue::Bool(b) => Some(*b),
        SochValue::Int(i) => Some(*i != 0),
        SochValue::Null => None,
        _ => None,
    }
}

/// Total ordering across SochValue for grouping/sorting.
pub fn compare_values(a: &SochValue, b: &SochValue) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (numeric(a), numeric(b)) {
        (Some(x), Some(y)) => return x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        _ => {}
    }
    match (a, b) {
        (SochValue::Text(x), SochValue::Text(y)) => x.cmp(y),
        (SochValue::Null, SochValue::Null) => Ordering::Equal,
        (SochValue::Null, _) => Ordering::Less,
        (_, SochValue::Null) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

/// Canonical hash representation of a group-key value.
/// Normalizes Int/UInt/Float-of-integral so `1`, `1u`, `1.0` group together.
fn key_repr(v: &SochValue) -> String {
    match v {
        SochValue::Null => "\u{0}N".to_string(),
        SochValue::Int(i) => format!("i{}", i),
        SochValue::UInt(u) => format!("i{}", u),
        SochValue::Float(f) => {
            if f.fract() == 0.0 && f.abs() < 9.0e15 {
                format!("i{}", *f as i64)
            } else {
                format!("f{}", f)
            }
        }
        SochValue::Text(s) => format!("s{}", s),
        SochValue::Bool(b) => format!("b{}", b),
        other => format!("{:?}", other),
    }
}

// ============================================================================
// Accumulators
// ============================================================================

#[derive(Debug)]
enum Acc {
    CountStar(u64),
    Count(u64),
    CountDistinct(HashSet<String>),
    /// Sum preserving integer-ness: (int_sum, float_sum, saw_float, saw_any)
    Sum {
        int: i64,
        float: f64,
        saw_float: bool,
        saw_any: bool,
        overflowed: bool,
    },
    Avg {
        sum: f64,
        n: u64,
    },
    Min(Option<SochValue>),
    Max(Option<SochValue>),
    Median(Vec<f64>),
    /// Welford online variance: (n, mean, m2)
    Stddev {
        n: u64,
        mean: f64,
        m2: f64,
    },
}

impl Acc {
    fn new(spec: &AggSpec) -> Self {
        match (spec.func, spec.arg.is_some(), spec.distinct) {
            (AggFn::Count, false, _) => Acc::CountStar(0),
            (AggFn::Count, true, true) => Acc::CountDistinct(HashSet::new()),
            (AggFn::Count, true, false) => Acc::Count(0),
            (AggFn::Sum, _, _) => Acc::Sum {
                int: 0,
                float: 0.0,
                saw_float: false,
                saw_any: false,
                overflowed: false,
            },
            (AggFn::Avg, _, _) => Acc::Avg { sum: 0.0, n: 0 },
            (AggFn::Min, _, _) => Acc::Min(None),
            (AggFn::Max, _, _) => Acc::Max(None),
            (AggFn::Median, _, _) => Acc::Median(Vec::new()),
            (AggFn::Stddev, _, _) => Acc::Stddev {
                n: 0,
                mean: 0.0,
                m2: 0.0,
            },
        }
    }

    /// Update with the evaluated argument value (`None` only for COUNT(*)).
    fn update(&mut self, val: Option<&SochValue>) {
        match self {
            Acc::CountStar(n) => *n += 1,
            Acc::Count(n) => {
                if let Some(v) = val {
                    if !v.is_null() {
                        *n += 1;
                    }
                }
            }
            Acc::CountDistinct(set) => {
                if let Some(v) = val {
                    if !v.is_null() {
                        set.insert(key_repr(v));
                    }
                }
            }
            Acc::Sum {
                int,
                float,
                saw_float,
                saw_any,
                overflowed,
            } => {
                let Some(v) = val else { return };
                match v {
                    SochValue::Int(i) => {
                        *saw_any = true;
                        match int.checked_add(*i) {
                            Some(s) => *int = s,
                            None => *overflowed = true,
                        }
                        *float += *i as f64;
                    }
                    SochValue::UInt(u) => {
                        *saw_any = true;
                        match int.checked_add(*u as i64) {
                            Some(s) => *int = s,
                            None => *overflowed = true,
                        }
                        *float += *u as f64;
                    }
                    SochValue::Float(f) => {
                        *saw_any = true;
                        *saw_float = true;
                        *float += *f;
                    }
                    _ => {}
                }
            }
            Acc::Avg { sum, n } => {
                if let Some(x) = val.and_then(numeric) {
                    *sum += x;
                    *n += 1;
                }
            }
            Acc::Min(cur) => {
                let Some(v) = val else { return };
                if v.is_null() {
                    return;
                }
                match cur {
                    None => *cur = Some(v.clone()),
                    Some(c) => {
                        if compare_values(v, c) == std::cmp::Ordering::Less {
                            *cur = Some(v.clone());
                        }
                    }
                }
            }
            Acc::Max(cur) => {
                let Some(v) = val else { return };
                if v.is_null() {
                    return;
                }
                match cur {
                    None => *cur = Some(v.clone()),
                    Some(c) => {
                        if compare_values(v, c) == std::cmp::Ordering::Greater {
                            *cur = Some(v.clone());
                        }
                    }
                }
            }
            Acc::Median(vals) => {
                if let Some(x) = val.and_then(numeric) {
                    vals.push(x);
                }
            }
            Acc::Stddev { n, mean, m2 } => {
                if let Some(x) = val.and_then(numeric) {
                    *n += 1;
                    let delta = x - *mean;
                    *mean += delta / *n as f64;
                    let delta2 = x - *mean;
                    *m2 += delta * delta2;
                }
            }
        }
    }

    /// Merge a partial accumulator (from a parallel chunk) into self.
    /// Both must originate from the same `AggSpec`.
    fn merge(&mut self, other: Acc) {
        match (self, other) {
            (Acc::CountStar(a), Acc::CountStar(b)) => *a += b,
            (Acc::Count(a), Acc::Count(b)) => *a += b,
            (Acc::CountDistinct(a), Acc::CountDistinct(b)) => a.extend(b),
            (
                Acc::Sum {
                    int,
                    float,
                    saw_float,
                    saw_any,
                    overflowed,
                },
                Acc::Sum {
                    int: i2,
                    float: f2,
                    saw_float: sf2,
                    saw_any: sa2,
                    overflowed: of2,
                },
            ) => {
                match int.checked_add(i2) {
                    Some(s) => *int = s,
                    None => *overflowed = true,
                }
                *float += f2;
                *saw_float |= sf2;
                *saw_any |= sa2;
                *overflowed |= of2;
            }
            (Acc::Avg { sum, n }, Acc::Avg { sum: s2, n: n2 }) => {
                *sum += s2;
                *n += n2;
            }
            (Acc::Min(a), Acc::Min(Some(b))) => match a {
                None => *a = Some(b),
                Some(cur) => {
                    if compare_values(&b, cur) == std::cmp::Ordering::Less {
                        *a = Some(b);
                    }
                }
            },
            (Acc::Max(a), Acc::Max(Some(b))) => match a {
                None => *a = Some(b),
                Some(cur) => {
                    if compare_values(&b, cur) == std::cmp::Ordering::Greater {
                        *a = Some(b);
                    }
                }
            },
            (Acc::Min(_), Acc::Min(None)) | (Acc::Max(_), Acc::Max(None)) => {}
            (Acc::Median(a), Acc::Median(b)) => a.extend(b),
            (
                Acc::Stddev { n, mean, m2 },
                Acc::Stddev {
                    n: nb,
                    mean: mb,
                    m2: m2b,
                },
            ) => {
                // Chan et al. parallel variance merge.
                if nb > 0 {
                    if *n == 0 {
                        *n = nb;
                        *mean = mb;
                        *m2 = m2b;
                    } else {
                        let na = *n as f64;
                        let nbf = nb as f64;
                        let delta = mb - *mean;
                        let total = na + nbf;
                        *mean += delta * nbf / total;
                        *m2 += m2b + delta * delta * na * nbf / total;
                        *n += nb;
                    }
                }
            }
            _ => unreachable!("mismatched accumulator merge"),
        }
    }

    fn finalize(self) -> SochValue {
        match self {
            Acc::CountStar(n) | Acc::Count(n) => SochValue::Int(n as i64),
            Acc::CountDistinct(set) => SochValue::Int(set.len() as i64),
            Acc::Sum {
                int,
                float,
                saw_float,
                saw_any,
                overflowed,
            } => {
                if !saw_any {
                    SochValue::Null
                } else if saw_float || overflowed {
                    SochValue::Float(float)
                } else {
                    SochValue::Int(int)
                }
            }
            Acc::Avg { sum, n } => {
                if n == 0 {
                    SochValue::Null
                } else {
                    SochValue::Float(sum / n as f64)
                }
            }
            Acc::Min(v) | Acc::Max(v) => v.unwrap_or(SochValue::Null),
            Acc::Median(mut vals) => {
                if vals.is_empty() {
                    return SochValue::Null;
                }
                let mid = vals.len() / 2;
                if vals.len() % 2 == 1 {
                    let (_, m, _) =
                        vals.select_nth_unstable_by(mid, |a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    SochValue::Float(*m)
                } else {
                    // Even count: average the two middle values.
                    let (lo, hi_first, _) =
                        vals.select_nth_unstable_by(mid, |a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    let lo_max = lo
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    SochValue::Float((lo_max + *hi_first) / 2.0)
                }
            }
            Acc::Stddev { n, m2, .. } => {
                if n < 2 {
                    SochValue::Null
                } else {
                    SochValue::Float((m2 / (n - 1) as f64).sqrt())
                }
            }
        }
    }
}

// ============================================================================
// The aggregation operator
// ============================================================================

struct GroupState {
    key_values: Vec<SochValue>,
    first_row: HashMap<String, SochValue>,
    accs: Vec<Acc>,
}

// ----------------------------------------------------------------------------
// Fast path: plain-column group keys and aggregate args
// ----------------------------------------------------------------------------

/// A group-key atom that borrows string data from the input rows —
/// zero allocations during accumulation lookups.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KeyAtom<'a> {
    Null,
    Int(i64),
    /// Normalized f64 bits (integral floats normalize to `Int`).
    FBits(u64),
    Str(&'a str),
    Bool(bool),
}

impl<'a> KeyAtom<'a> {
    fn from_value(v: &'a SochValue) -> Self {
        match v {
            SochValue::Null => KeyAtom::Null,
            SochValue::Int(i) => KeyAtom::Int(*i),
            SochValue::UInt(u) => KeyAtom::Int(*u as i64),
            SochValue::Float(f) => {
                if f.fract() == 0.0 && f.abs() < 9.0e15 {
                    KeyAtom::Int(*f as i64)
                } else if f.is_nan() {
                    KeyAtom::FBits(f64::NAN.to_bits())
                } else {
                    KeyAtom::FBits(f.to_bits())
                }
            }
            SochValue::Text(s) => KeyAtom::Str(s.as_str()),
            SochValue::Bool(b) => KeyAtom::Bool(*b),
            _ => KeyAtom::Null,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupKey<'a> {
    Empty,
    One(KeyAtom<'a>),
    Many(Vec<KeyAtom<'a>>),
}

static NULL_VALUE: SochValue = SochValue::Null;

/// Resolve a column reference against a row, trying qualified name first.
#[inline]
fn col_get<'r>(row: &'r HashMap<String, SochValue>, col: &PlainCol) -> &'r SochValue {
    if let Some(q) = &col.qualified {
        if let Some(v) = row.get(q) {
            return v;
        }
    }
    row.get(&col.name).unwrap_or(&NULL_VALUE)
}

/// Pre-resolved plain column: unqualified name + optional "table.col" form.
struct PlainCol {
    name: String,
    qualified: Option<String>,
}

fn as_plain_col(expr: &Expr) -> Option<PlainCol> {
    match expr {
        Expr::Column(c) => Some(PlainCol {
            name: c.column.clone(),
            qualified: c.table.as_ref().map(|t| format!("{}.{}", t, c.column)),
        }),
        _ => None,
    }
}

/// Build the borrowed group key for one row.
fn make_group_key<'r>(
    row: &'r HashMap<String, SochValue>,
    group_cols: &[PlainCol],
) -> GroupKey<'r> {
    match group_cols.len() {
        0 => GroupKey::Empty,
        1 => GroupKey::One(KeyAtom::from_value(col_get(row, &group_cols[0]))),
        _ => GroupKey::Many(
            group_cols
                .iter()
                .map(|c| KeyAtom::from_value(col_get(row, c)))
                .collect(),
        ),
    }
}

/// Try the optimized accumulation path. Applicable when every GROUP BY
/// expression and every aggregate argument is a plain column reference
/// (which covers typical analytics queries). Returns group states in
/// first-seen order (per-chunk order under parallel execution).
fn accumulate_fast<'a>(
    select: &SelectStmt,
    specs: &[AggSpec],
    rows: &'a [HashMap<String, SochValue>],
) -> Option<Vec<GroupState>> {
    // Pre-resolve group-key columns.
    let group_cols: Vec<PlainCol> = select
        .group_by
        .iter()
        .map(as_plain_col)
        .collect::<Option<Vec<_>>>()?;
    // Pre-resolve aggregate argument columns (None = COUNT(*)).
    let arg_cols: Vec<Option<PlainCol>> = specs
        .iter()
        .map(|s| match &s.arg {
            None => Some(None),
            Some(e) => as_plain_col(e).map(Some),
        })
        .collect::<Option<Vec<_>>>()?;

    let accumulate_chunk = |chunk: &'a [HashMap<String, SochValue>]| -> Vec<(GroupKey<'a>, GroupState)> {
        let mut order: Vec<(GroupKey<'a>, GroupState)> = Vec::new();
        let mut index: HashMap<GroupKey<'a>, usize> = HashMap::new();
        for row in chunk {
            let key = make_group_key(row, &group_cols);
            let idx = match index.get(&key) {
                Some(&i) => i,
                None => {
                    let state = GroupState {
                        key_values: group_cols
                            .iter()
                            .map(|c| col_get(row, c).clone())
                            .collect(),
                        first_row: row.clone(),
                        accs: specs.iter().map(Acc::new).collect(),
                    };
                    order.push((key.clone(), state));
                    index.insert(key, order.len() - 1);
                    order.len() - 1
                }
            };
            let accs = &mut order[idx].1.accs;
            for (acc, arg) in accs.iter_mut().zip(arg_cols.iter()) {
                match arg {
                    None => acc.update(None),
                    Some(col) => acc.update(Some(col_get(row, col))),
                }
            }
        }
        order
    };

    let merged: Vec<(GroupKey<'a>, GroupState)> = if rows.len() >= PARALLEL_THRESHOLD {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = (rows.len() / (n_threads * 4)).max(16_384);
        let partials: Vec<Vec<(GroupKey<'a>, GroupState)>> = rows
            .par_chunks(chunk_size)
            .map(accumulate_chunk)
            .collect();
        // Merge chunk partials in chunk order.
        let mut order: Vec<(GroupKey<'a>, GroupState)> = Vec::new();
        let mut index: HashMap<GroupKey<'a>, usize> = HashMap::new();
        for partial in partials {
            for (key, state) in partial {
                match index.get(&key) {
                    Some(&i) => {
                        let dst = &mut order[i].1;
                        for (a, b) in dst.accs.iter_mut().zip(state.accs.into_iter()) {
                            a.merge(b);
                        }
                    }
                    None => {
                        order.push((key.clone(), state));
                        index.insert(key, order.len() - 1);
                    }
                }
            }
        }
        order
    } else {
        accumulate_chunk(rows)
    };

    Some(merged.into_iter().map(|(_, s)| s).collect())
}

/// Execute aggregation over materialized input rows (already WHERE-filtered).
///
/// Handles GROUP BY, all aggregate accumulation, HAVING, ORDER BY,
/// OFFSET/LIMIT, and final projection. Returns `ExecutionResult::Rows`.
pub fn execute_aggregate(
    select: &SelectStmt,
    rows: &[HashMap<String, SochValue>],
    params: &[SochValue],
    limit: Option<usize>,
    offset: Option<usize>,
) -> SqlResult<ExecutionResult> {
    let specs = collect_agg_specs(select);
    let grouped = !select.group_by.is_empty();

    // ---- accumulate ----
    // Fast path: plain-column keys/args, borrowed-key hashing, parallel
    // partitioned accumulation. Falls back to the general expression-based
    // path for computed keys or computed aggregate arguments.
    let mut order: Vec<GroupState> = match accumulate_fast(select, &specs, rows) {
        Some(states) => states,
        None => {
            let mut order: Vec<GroupState> = Vec::new();
            let mut index: HashMap<Vec<String>, usize> = HashMap::new();

            for row in rows {
                let key_values: Vec<SochValue> = select
                    .group_by
                    .iter()
                    .map(|e| eval_scalar(e, row, params))
                    .collect();
                let hash_key: Vec<String> = key_values.iter().map(key_repr).collect();

                let idx = match index.get(&hash_key) {
                    Some(&i) => i,
                    None => {
                        let state = GroupState {
                            key_values,
                            first_row: row.clone(),
                            accs: specs.iter().map(Acc::new).collect(),
                        };
                        order.push(state);
                        index.insert(hash_key, order.len() - 1);
                        order.len() - 1
                    }
                };

                let state = &mut order[idx];
                for (acc, spec) in state.accs.iter_mut().zip(specs.iter()) {
                    match &spec.arg {
                        None => acc.update(None),
                        Some(arg) => {
                            let v = eval_scalar(arg, row, params);
                            acc.update(Some(&v));
                        }
                    }
                }
            }
            order
        }
    };

    // Ungrouped aggregate over zero rows still yields one (empty) group.
    if !grouped && order.is_empty() {
        order.push(GroupState {
            key_values: Vec::new(),
            first_row: HashMap::new(),
            accs: specs.iter().map(Acc::new).collect(),
        });
    }

    // ---- finalize: synthesize one row per group ----
    let group_names: Vec<String> = select.group_by.iter().map(render_expr_name).collect();

    let mut out_rows: Vec<HashMap<String, SochValue>> = Vec::with_capacity(order.len());
    for state in order {
        // Start from the first row of the group so non-aggregate columns
        // (lenient mode) and qualified names still resolve.
        let mut row = state.first_row;
        for (name, val) in group_names.iter().zip(state.key_values.into_iter()) {
            row.insert(name.clone(), val);
        }
        for (spec, acc) in specs.iter().zip(state.accs.into_iter()) {
            row.insert(spec.key.clone(), acc.finalize());
        }
        out_rows.push(row);
    }

    // ---- HAVING ----
    if let Some(having) = &select.having {
        out_rows.retain(|row| {
            matches!(eval_scalar(having, row, params), SochValue::Bool(true))
        });
    }

    // ---- ORDER BY (may reference aggregates or aliases) ----
    if !select.order_by.is_empty() {
        // Bind aliases so ORDER BY alias works.
        let alias_map: Vec<(String, Expr)> = select
            .columns
            .iter()
            .filter_map(|item| match item {
                SelectItem::Expr {
                    expr,
                    alias: Some(a),
                } => Some((a.clone(), expr.clone())),
                _ => None,
            })
            .collect();
        for row in &mut out_rows {
            for (alias, expr) in &alias_map {
                if !row.contains_key(alias) {
                    let v = eval_scalar(expr, row, params);
                    row.insert(alias.clone(), v);
                }
            }
        }
        out_rows.sort_by(|a, b| {
            for item in &select.order_by {
                let va = eval_scalar(&item.expr, a, params);
                let vb = eval_scalar(&item.expr, b, params);
                let mut cmp = compare_values(&va, &vb);
                if !item.asc {
                    cmp = cmp.reverse();
                }
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    // ---- OFFSET / LIMIT ----
    if let Some(off) = offset {
        if off > 0 {
            out_rows.drain(..off.min(out_rows.len()));
        }
    }
    if let Some(lim) = limit {
        out_rows.truncate(lim);
    }

    // ---- projection ----
    let mut columns: Vec<String> = Vec::new();
    let mut projections: Vec<(String, Expr)> = Vec::new();
    for item in &select.columns {
        match item {
            SelectItem::Wildcard | SelectItem::QualifiedWildcard(_) => {
                // SELECT * with GROUP BY: project group keys then aggregates.
                for name in &group_names {
                    columns.push(name.clone());
                    projections.push((
                        name.clone(),
                        Expr::Column(ColumnRef::new(name.clone())),
                    ));
                }
                for spec in &specs {
                    columns.push(spec.key.clone());
                    projections.push((
                        spec.key.clone(),
                        Expr::Column(ColumnRef::new(spec.key.clone())),
                    ));
                }
            }
            SelectItem::Expr { expr, alias } => {
                let name = alias.clone().unwrap_or_else(|| render_expr_name(expr));
                columns.push(name.clone());
                projections.push((name, expr.clone()));
            }
        }
    }

    let projected: Vec<HashMap<String, SochValue>> = out_rows
        .into_iter()
        .map(|row| {
            let mut out = HashMap::with_capacity(projections.len());
            for (name, expr) in &projections {
                let v = eval_scalar(expr, &row, params);
                out.insert(name.clone(), v);
            }
            out
        })
        .collect();

    Ok(ExecutionResult::Rows {
        columns,
        rows: projected,
    })
}

#[cfg(test)]
mod tests {
    use super::super::bridge::{SqlBridge, SqlConnection};
    use super::*;

    fn fcall(name: &str, arg: &str) -> Expr {
        Expr::Function(FunctionCall {
            name: ObjectName::new(name),
            args: vec![Expr::Column(ColumnRef::new(arg))],
            distinct: false,
            filter: None,
            over: None,
        })
    }

    #[test]
    fn agg_fn_recognition() {
        assert_eq!(AggFn::from_name("median"), Some(AggFn::Median));
        assert_eq!(AggFn::from_name("STDDEV"), Some(AggFn::Stddev));
        assert_eq!(AggFn::from_name("stddev_samp"), Some(AggFn::Stddev));
        assert_eq!(AggFn::from_name("upper"), None);
    }

    #[test]
    fn canonical_keys() {
        assert_eq!(render_expr_name(&fcall("SUM", "v1")), "sum(v1)");
        assert_eq!(render_expr_name(&fcall("Median", "v3")), "median(v3)");
    }

    // ========================================================================
    // End-to-end SQL tests through SqlBridge with an in-memory connection
    // ========================================================================

    /// In-memory table store implementing SqlConnection for tests.
    struct DataConn {
        tables: HashMap<String, Vec<HashMap<String, SochValue>>>,
    }

    impl DataConn {
        fn new() -> Self {
            Self {
                tables: HashMap::new(),
            }
        }

        fn with_table(
            mut self,
            name: &str,
            cols: &[&str],
            rows: Vec<Vec<SochValue>>,
        ) -> Self {
            let rows = rows
                .into_iter()
                .map(|vals| {
                    cols.iter()
                        .map(|c| c.to_string())
                        .zip(vals.into_iter())
                        .collect::<HashMap<_, _>>()
                })
                .collect();
            self.tables.insert(name.to_string(), rows);
            self
        }
    }

    impl SqlConnection for DataConn {
        fn select(
            &self,
            table: &str,
            _: &[String],
            _where_clause: Option<&Expr>,
            _: &[OrderByItem],
            _: Option<usize>,
            _: Option<usize>,
            _: &[SochValue],
        ) -> SqlResult<ExecutionResult> {
            // Tests using the aggregate path don't push WHERE here.
            let rows = self.tables.get(table).cloned().unwrap_or_default();
            Ok(ExecutionResult::Rows {
                columns: vec![],
                rows,
            })
        }
        fn insert(
            &mut self,
            _: &str,
            _: Option<&[String]>,
            _: &[Vec<Expr>],
            _: Option<&OnConflict>,
            _: &[SochValue],
        ) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::RowsAffected(0))
        }
        fn update(
            &mut self,
            _: &str,
            _: &[Assignment],
            _: Option<&Expr>,
            _: &[SochValue],
        ) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::RowsAffected(0))
        }
        fn delete(
            &mut self,
            _: &str,
            _: Option<&Expr>,
            _: &[SochValue],
        ) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::RowsAffected(0))
        }
        fn create_table(&mut self, _: &CreateTableStmt) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::Ok)
        }
        fn drop_table(&mut self, _: &DropTableStmt) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::Ok)
        }
        fn create_index(&mut self, _: &CreateIndexStmt) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::Ok)
        }
        fn drop_index(&mut self, _: &DropIndexStmt) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::Ok)
        }
        fn alter_table(&mut self, _: &AlterTableStmt) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::Ok)
        }
        fn begin(&mut self, _: &BeginStmt) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::TransactionOk)
        }
        fn commit(&mut self) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::TransactionOk)
        }
        fn rollback(&mut self, _: Option<&str>) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::TransactionOk)
        }
        fn table_exists(&self, t: &str) -> SqlResult<bool> {
            Ok(self.tables.contains_key(t))
        }
        fn index_exists(&self, _: &str) -> SqlResult<bool> {
            Ok(false)
        }
        fn scan_all(
            &self,
            table: &str,
            _: &[String],
        ) -> SqlResult<Vec<HashMap<String, SochValue>>> {
            Ok(self.tables.get(table).cloned().unwrap_or_default())
        }
        fn eval_join_predicate(
            &self,
            expr: &Expr,
            row: &HashMap<String, SochValue>,
            params: &[SochValue],
        ) -> Option<bool> {
            match eval_scalar(expr, row, params) {
                SochValue::Bool(b) => Some(b),
                SochValue::Null => Some(false),
                _ => None,
            }
        }
    }

    fn i(v: i64) -> SochValue {
        SochValue::Int(v)
    }
    fn f(v: f64) -> SochValue {
        SochValue::Float(v)
    }
    fn t(v: &str) -> SochValue {
        SochValue::Text(v.to_string())
    }

    /// db-benchmark-shaped fixture: x(id1 text, id3 text, v1 int, v2 int, v3 float)
    fn bench_bridge() -> SqlBridge<DataConn> {
        let conn = DataConn::new().with_table(
            "x",
            &["id1", "id3", "v1", "v2", "v3"],
            vec![
                vec![t("id001"), t("id0000001"), i(1), i(10), f(1.0)],
                vec![t("id001"), t("id0000002"), i(2), i(20), f(2.0)],
                vec![t("id002"), t("id0000001"), i(3), i(30), f(3.0)],
                vec![t("id002"), t("id0000002"), i(4), i(40), f(4.0)],
            ],
        );
        SqlBridge::new(conn)
    }

    fn rows_of(result: ExecutionResult) -> Vec<HashMap<String, SochValue>> {
        match result {
            ExecutionResult::Rows { rows, .. } => rows,
            other => panic!("expected rows, got {:?}", other),
        }
    }

    fn get<'a>(row: &'a HashMap<String, SochValue>, k: &str) -> &'a SochValue {
        row.get(k)
            .unwrap_or_else(|| panic!("column '{}' missing from {:?}", k, row))
    }

    #[test]
    fn groupby_sum_q1_shape() {
        // db-benchmark q1: SELECT id1, sum(v1) AS v1 FROM x GROUP BY id1
        let mut b = bench_bridge();
        let rows = rows_of(
            b.execute("SELECT id1, sum(v1) AS v1 FROM x GROUP BY id1 ORDER BY id1")
                .unwrap(),
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(get(&rows[0], "id1"), &t("id001"));
        assert_eq!(get(&rows[0], "v1"), &i(3));
        assert_eq!(get(&rows[1], "id1"), &t("id002"));
        assert_eq!(get(&rows[1], "v1"), &i(7));
    }

    #[test]
    fn groupby_multi_key_mean() {
        // q4-like: SELECT id1, id3, avg(v1) AS m FROM x GROUP BY id1, id3
        let mut b = bench_bridge();
        let rows = rows_of(
            b.execute(
                "SELECT id1, id3, avg(v1) AS m FROM x GROUP BY id1, id3 ORDER BY id1, id3",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), 4);
        assert_eq!(get(&rows[0], "m"), &f(1.0));
        assert_eq!(get(&rows[3], "m"), &f(4.0));
    }

    #[test]
    fn median_and_stddev() {
        // q7-like: SELECT median(v3), stddev(v3) FROM x
        // v3 = [1,2,3,4]: median = 2.5, sample sd = sqrt(5/3)
        let mut b = bench_bridge();
        let rows = rows_of(
            b.execute("SELECT median(v3) AS med, stddev(v3) AS sd FROM x")
                .unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(get(&rows[0], "med"), &f(2.5));
        match get(&rows[0], "sd") {
            SochValue::Float(sd) => {
                assert!((sd - (5.0f64 / 3.0).sqrt()).abs() < 1e-12, "sd={}", sd)
            }
            other => panic!("expected float sd, got {:?}", other),
        }
    }

    #[test]
    fn median_odd_count() {
        let conn = DataConn::new().with_table(
            "t",
            &["v"],
            vec![vec![f(5.0)], vec![f(1.0)], vec![f(3.0)]],
        );
        let mut b = SqlBridge::new(conn);
        let rows = rows_of(b.execute("SELECT median(v) AS m FROM t").unwrap());
        assert_eq!(get(&rows[0], "m"), &f(3.0));
    }

    #[test]
    fn range_expression_q9_shape() {
        // q9: SELECT id3, max(v1) - min(v2) AS range_v1_v2 FROM x GROUP BY id3
        let mut b = bench_bridge();
        let rows = rows_of(
            b.execute(
                "SELECT id3, max(v1) - min(v2) AS range_v1_v2 FROM x GROUP BY id3 ORDER BY id3",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), 2);
        // id0000001: max(v1)=3, min(v2)=10 -> -7
        assert_eq!(get(&rows[0], "range_v1_v2"), &i(-7));
        // id0000002: max(v1)=4, min(v2)=20 -> -16
        assert_eq!(get(&rows[1], "range_v1_v2"), &i(-16));
    }

    #[test]
    fn count_star_vs_count_col_with_nulls() {
        let conn = DataConn::new().with_table(
            "t",
            &["g", "v"],
            vec![
                vec![t("a"), i(1)],
                vec![t("a"), SochValue::Null],
                vec![t("b"), i(2)],
            ],
        );
        let mut b = SqlBridge::new(conn);
        let rows = rows_of(
            b.execute(
                "SELECT g, count(*) AS n, count(v) AS nv FROM t GROUP BY g ORDER BY g",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(get(&rows[0], "n"), &i(2));
        assert_eq!(get(&rows[0], "nv"), &i(1));
        assert_eq!(get(&rows[1], "n"), &i(1));
        assert_eq!(get(&rows[1], "nv"), &i(1));
    }

    #[test]
    fn count_distinct() {
        // q6-like: SELECT id3, count(DISTINCT id1) AS u FROM x GROUP BY id3
        let mut b = bench_bridge();
        let rows = rows_of(
            b.execute("SELECT id3, count(DISTINCT id1) AS u FROM x GROUP BY id3 ORDER BY id3")
                .unwrap(),
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(get(&rows[0], "u"), &i(2));
        assert_eq!(get(&rows[1], "u"), &i(2));
    }

    #[test]
    fn having_filters_groups() {
        let mut b = bench_bridge();
        let rows = rows_of(
            b.execute("SELECT id1, sum(v1) AS s FROM x GROUP BY id1 HAVING sum(v1) > 5")
                .unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(get(&rows[0], "id1"), &t("id002"));
        assert_eq!(get(&rows[0], "s"), &i(7));
    }

    #[test]
    fn order_by_aggregate_desc_with_limit() {
        let mut b = bench_bridge();
        let rows = rows_of(
            b.execute("SELECT id1, sum(v1) AS s FROM x GROUP BY id1 ORDER BY s DESC LIMIT 1")
                .unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(get(&rows[0], "id1"), &t("id002"));
    }

    #[test]
    fn ungrouped_aggregate_over_empty_table() {
        let conn = DataConn::new().with_table("e", &["v"], vec![]);
        let mut b = SqlBridge::new(conn);
        let rows = rows_of(
            b.execute("SELECT count(*) AS n, sum(v) AS s FROM e").unwrap(),
        );
        assert_eq!(rows.len(), 1, "ungrouped agg over empty input = one row");
        assert_eq!(get(&rows[0], "n"), &i(0));
        assert_eq!(get(&rows[0], "s"), &SochValue::Null);
    }

    #[test]
    fn grouped_aggregate_over_empty_table_yields_no_rows() {
        let conn = DataConn::new().with_table("e", &["g", "v"], vec![]);
        let mut b = SqlBridge::new(conn);
        let rows = rows_of(
            b.execute("SELECT g, sum(v) AS s FROM e GROUP BY g").unwrap(),
        );
        assert!(rows.is_empty());
    }

    #[test]
    fn sum_overflow_promotes_to_float() {
        let conn = DataConn::new().with_table(
            "t",
            &["v"],
            vec![vec![i(i64::MAX)], vec![i(i64::MAX)]],
        );
        let mut b = SqlBridge::new(conn);
        let rows = rows_of(b.execute("SELECT sum(v) AS s FROM t").unwrap());
        match get(&rows[0], "s") {
            SochValue::Float(v) => assert!(*v > 1.8e19),
            other => panic!("expected float after overflow, got {:?}", other),
        }
    }

    #[test]
    fn aggregate_after_join() {
        // join + group: SELECT x.id1, sum(y.w) FROM x JOIN y ON x.id1 = y.id1 GROUP BY x.id1
        let conn = DataConn::new()
            .with_table(
                "a",
                &["id", "v"],
                vec![
                    vec![t("k1"), i(1)],
                    vec![t("k1"), i(2)],
                    vec![t("k2"), i(3)],
                ],
            )
            .with_table(
                "b",
                &["id", "w"],
                vec![vec![t("k1"), i(10)], vec![t("k2"), i(20)]],
            );
        let mut br = SqlBridge::new(conn);
        let rows = rows_of(
            br.execute(
                "SELECT a.id, sum(a.v) AS sv, sum(b.w) AS sw \
                 FROM a JOIN b ON a.id = b.id GROUP BY a.id ORDER BY a.id",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(get(&rows[0], "sv"), &i(3));
        assert_eq!(get(&rows[0], "sw"), &i(20)); // 10 joined to both k1 rows
        assert_eq!(get(&rows[1], "sv"), &i(3));
        assert_eq!(get(&rows[1], "sw"), &i(20));
    }

    #[test]
    fn lowercase_function_names_parse() {
        // db-benchmark SQL uses lowercase: sum(v1), median(v3)
        let mut b = bench_bridge();
        assert!(b.execute("SELECT id1, sum(v1) FROM x GROUP BY id1").is_ok());
        assert!(b.execute("SELECT median(v3) FROM x").is_ok());
        assert!(b.execute("SELECT stddev(v3) FROM x").is_ok());
    }

    #[test]
    fn parallel_path_matches_reference_computation() {
        // 150k rows (> PARALLEL_THRESHOLD) exercising the rayon merge:
        // sum, avg, count, median, stddev per group, verified against
        // values computed directly in the test.
        let n: usize = 150_000;
        let groups = 7usize;
        let mut data: Vec<Vec<SochValue>> = Vec::with_capacity(n);
        for idx in 0..n {
            data.push(vec![
                t(&format!("g{}", idx % groups)),
                f((idx * 31 % 1000) as f64 / 4.0),
            ]);
        }
        // Reference computation.
        let mut per_group: Vec<Vec<f64>> = vec![Vec::new(); groups];
        for idx in 0..n {
            per_group[idx % groups].push((idx * 31 % 1000) as f64 / 4.0);
        }

        let conn = DataConn::new().with_table("big", &["g", "v"], data);
        let mut b = SqlBridge::new(conn);
        let rows = rows_of(
            b.execute(
                "SELECT g, count(*) AS n, sum(v) AS s, avg(v) AS m, \
                 median(v) AS med, stddev(v) AS sd FROM big GROUP BY g ORDER BY g",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), groups);

        for (gi, row) in rows.iter().enumerate() {
            let vals = &per_group[gi];
            let cnt = vals.len() as f64;
            let sum: f64 = vals.iter().sum();
            let mean = sum / cnt;
            let var =
                vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (cnt - 1.0);
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = if sorted.len() % 2 == 1 {
                sorted[sorted.len() / 2]
            } else {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            };

            assert_eq!(get(row, "g"), &t(&format!("g{}", gi)));
            assert_eq!(get(row, "n"), &i(vals.len() as i64));
            match get(row, "s") {
                SochValue::Float(v) => assert!((v - sum).abs() < 1e-6, "sum"),
                other => panic!("sum type {:?}", other),
            }
            match get(row, "m") {
                SochValue::Float(v) => assert!((v - mean).abs() < 1e-9, "mean"),
                other => panic!("mean type {:?}", other),
            }
            match get(row, "med") {
                SochValue::Float(v) => assert!((v - med).abs() < 1e-9, "median"),
                other => panic!("median type {:?}", other),
            }
            match get(row, "sd") {
                SochValue::Float(v) => {
                    assert!((v - var.sqrt()).abs() < 1e-9, "sd {} vs {}", v, var.sqrt())
                }
                other => panic!("sd type {:?}", other),
            }
        }
    }

    #[test]
    fn unaliased_aggregate_column_name_is_canonical() {
        let mut b = bench_bridge();
        let result = b.execute("SELECT id1, sum(v1) FROM x GROUP BY id1").unwrap();
        let cols = result.columns().unwrap().clone();
        assert!(cols.contains(&"sum(v1)".to_string()), "cols={:?}", cols);
    }
}

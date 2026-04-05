// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hash aggregate operator (GROUP BY + aggregate functions).
//!
//! Supports: COUNT, SUM, AVG, MIN, MAX, COUNT(DISTINCT ...)

use crate::soch_ql::SochValue;
use crate::sql::ast::Expr;
use super::eval::{eval_expr, compare_values};
use super::node::PlanNode;
use super::types::{Row, Schema, ColumnMeta};
use sochdb_core::Result;
use std::collections::HashMap;

/// Aggregate function types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggFunc {
    Count,
    CountDistinct,
    Sum,
    Avg,
    Min,
    Max,
}

/// Definition of an aggregate computation.
#[derive(Debug, Clone)]
pub struct AggDef {
    /// Function type.
    pub func: AggFunc,
    /// Expression to aggregate (None for COUNT(*)).
    pub expr: Option<Expr>,
    /// Output column alias.
    pub alias: String,
}

/// Internal accumulator for one aggregate function.
struct Accumulator {
    func: AggFunc,
    count: u64,
    sum_int: i64,
    sum_float: f64,
    is_float: bool,
    min_val: Option<SochValue>,
    max_val: Option<SochValue>,
    distinct_set: Option<Vec<SochValue>>,
}

impl Accumulator {
    fn new(func: &AggFunc) -> Self {
        Self {
            func: func.clone(),
            count: 0,
            sum_int: 0,
            sum_float: 0.0,
            is_float: false,
            min_val: None,
            max_val: None,
            distinct_set: if matches!(func, AggFunc::CountDistinct) {
                Some(Vec::new())
            } else {
                None
            },
        }
    }

    fn accumulate(&mut self, val: &SochValue) {
        // Skip NULLs for all aggregates except COUNT(*)
        if matches!(val, SochValue::Null) {
            // COUNT(*) still counts NULLs — handled by the caller
            // passing SochValue::Bool(true) for count(*)
            if matches!(self.func, AggFunc::Count) {
                // COUNT(*) — caller decides, regular COUNT(col) should skip
                return;
            }
            return;
        }

        match self.func {
            AggFunc::Count => {
                self.count += 1;
            }
            AggFunc::CountDistinct => {
                if let Some(set) = &mut self.distinct_set {
                    let already = set.iter().any(|v| {
                        compare_values(v, val) == Some(std::cmp::Ordering::Equal)
                    });
                    if !already {
                        set.push(val.clone());
                    }
                }
            }
            AggFunc::Sum => {
                match val {
                    SochValue::Int(i) => {
                        if self.is_float {
                            self.sum_float += *i as f64;
                        } else {
                            self.sum_int += i;
                        }
                    }
                    SochValue::UInt(u) => {
                        if self.is_float {
                            self.sum_float += *u as f64;
                        } else {
                            self.sum_int += *u as i64;
                        }
                    }
                    SochValue::Float(f) => {
                        if !self.is_float {
                            self.sum_float = self.sum_int as f64;
                            self.is_float = true;
                        }
                        self.sum_float += f;
                    }
                    _ => {}
                }
                self.count += 1;
            }
            AggFunc::Avg => {
                match val {
                    SochValue::Int(i) => self.sum_float += *i as f64,
                    SochValue::UInt(u) => self.sum_float += *u as f64,
                    SochValue::Float(f) => self.sum_float += f,
                    _ => {}
                }
                self.count += 1;
            }
            AggFunc::Min => {
                let update = match &self.min_val {
                    None => true,
                    Some(current) => {
                        compare_values(val, current) == Some(std::cmp::Ordering::Less)
                    }
                };
                if update {
                    self.min_val = Some(val.clone());
                }
            }
            AggFunc::Max => {
                let update = match &self.max_val {
                    None => true,
                    Some(current) => {
                        compare_values(val, current) == Some(std::cmp::Ordering::Greater)
                    }
                };
                if update {
                    self.max_val = Some(val.clone());
                }
            }
        }
    }

    fn finalize(&self) -> SochValue {
        match self.func {
            AggFunc::Count => SochValue::Int(self.count as i64),
            AggFunc::CountDistinct => {
                SochValue::Int(
                    self.distinct_set.as_ref().map_or(0, |s| s.len()) as i64,
                )
            }
            AggFunc::Sum => {
                if self.count == 0 {
                    SochValue::Null
                } else if self.is_float {
                    SochValue::Float(self.sum_float)
                } else {
                    SochValue::Int(self.sum_int)
                }
            }
            AggFunc::Avg => {
                if self.count == 0 {
                    SochValue::Null
                } else {
                    SochValue::Float(self.sum_float / self.count as f64)
                }
            }
            AggFunc::Min => self.min_val.clone().unwrap_or(SochValue::Null),
            AggFunc::Max => self.max_val.clone().unwrap_or(SochValue::Null),
        }
    }
}

/// Hash-key for GROUP BY: tuple of group-by values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupKey(Vec<GroupVal>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupVal {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Text(String),
    Other(String),
}

impl From<&SochValue> for GroupVal {
    fn from(v: &SochValue) -> Self {
        match v {
            SochValue::Null => GroupVal::Null,
            SochValue::Bool(b) => GroupVal::Bool(*b),
            SochValue::Int(i) => GroupVal::Int(*i),
            SochValue::UInt(u) => GroupVal::UInt(*u),
            SochValue::Text(s) => GroupVal::Text(s.clone()),
            other => GroupVal::Other(format!("{:?}", other)),
        }
    }
}

/// Group state: group-by values + accumulators.
struct GroupState {
    key_values: Vec<SochValue>,
    accumulators: Vec<Accumulator>,
}

/// Hash aggregate operator.
///
/// Materializes all input, groups by key expressions, computes aggregates,
/// then emits one row per group.
///
/// ```text
/// HashAggregate(group_by=[dept], aggs=[COUNT(*), AVG(salary)])
///   └── input
/// ```
pub struct HashAggregateNode {
    input: Box<dyn PlanNode>,
    group_by_exprs: Vec<Expr>,
    agg_defs: Vec<AggDef>,
    output_schema: Schema,
    /// Materialized groups (lazily computed).
    groups: Option<Vec<Row>>,
    pos: usize,
    /// Whether this is a global aggregate (no GROUP BY).
    is_global: bool,
}

impl HashAggregateNode {
    pub fn new(
        input: Box<dyn PlanNode>,
        group_by_exprs: Vec<Expr>,
        agg_defs: Vec<AggDef>,
    ) -> Self {
        let is_global = group_by_exprs.is_empty();

        // Build output schema: group-by columns + aggregate columns
        let mut cols: Vec<ColumnMeta> = group_by_exprs
            .iter()
            .map(|e| {
                let name = match e {
                    Expr::Column(c) => c.column.clone(),
                    _ => format!("{:?}", e),
                };
                ColumnMeta::new(name)
            })
            .collect();
        for ad in &agg_defs {
            cols.push(ColumnMeta::new(ad.alias.clone()));
        }
        let output_schema = Schema::new(cols);

        Self {
            input,
            group_by_exprs,
            agg_defs,
            output_schema,
            groups: None,
            pos: 0,
            is_global,
        }
    }

    fn materialize(&mut self) -> Result<()> {
        if self.groups.is_some() {
            return Ok(());
        }

        let input_schema = self.input.schema().clone();
        let mut group_map: HashMap<GroupKey, GroupState> = HashMap::new();
        let mut group_order: Vec<GroupKey> = Vec::new(); // Preserve insertion order

        // Count(*) tracking
        let has_count_star: Vec<bool> = self.agg_defs.iter().map(|ad| {
            matches!(ad.func, AggFunc::Count) && ad.expr.is_none()
        }).collect();

        while let Some(row) = self.input.next()? {
            // Evaluate group-by keys
            let key_values: Vec<SochValue> = self
                .group_by_exprs
                .iter()
                .map(|e| eval_expr(e, &row, &input_schema).unwrap_or(SochValue::Null))
                .collect();

            let group_key = GroupKey(key_values.iter().map(GroupVal::from).collect());

            let state = group_map.entry(group_key.clone()).or_insert_with(|| {
                group_order.push(group_key.clone());
                GroupState {
                    key_values: key_values.clone(),
                    accumulators: self
                        .agg_defs
                        .iter()
                        .map(|ad| Accumulator::new(&ad.func))
                        .collect(),
                }
            });

            // Accumulate values
            for (i, ad) in self.agg_defs.iter().enumerate() {
                if has_count_star[i] {
                    // COUNT(*) — count every row
                    state.accumulators[i].count += 1;
                } else if let Some(expr) = &ad.expr {
                    let val = eval_expr(expr, &row, &input_schema)?;
                    state.accumulators[i].accumulate(&val);
                }
            }
        }

        // Handle global aggregate with no input rows
        if self.is_global && group_map.is_empty() {
            let mut row: Row = Vec::new();
            for ad in &self.agg_defs {
                let acc = Accumulator::new(&ad.func);
                row.push(acc.finalize());
            }
            self.groups = Some(vec![row]);
            return Ok(());
        }

        // Build output rows in insertion order
        let mut result = Vec::with_capacity(group_order.len());
        for gk in &group_order {
            if let Some(state) = group_map.get(gk) {
                let mut row: Row = state.key_values.clone();
                for acc in &state.accumulators {
                    row.push(acc.finalize());
                }
                result.push(row);
            }
        }

        self.groups = Some(result);
        Ok(())
    }
}

impl PlanNode for HashAggregateNode {
    fn schema(&self) -> &Schema {
        &self.output_schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        self.materialize()?;

        if let Some(groups) = &self.groups {
            if self.pos < groups.len() {
                let row = groups[self.pos].clone();
                self.pos += 1;
                Ok(Some(row))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.groups = None;
        self.pos = 0;
        self.input.reset()
    }
}

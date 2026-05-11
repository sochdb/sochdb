// SPDX-License-Identifier: AGPL-3.0-or-later

//! SQL expression evaluator over Volcano rows.
//!
//! Evaluates [`sql::ast::Expr`] nodes against a `(Row, Schema)` context,
//! producing [`SochValue`] results. Used by Filter, Project, Sort, and
//! Join operators for predicate and expression evaluation.

use crate::soch_ql::SochValue;
use crate::sql::ast::{BinaryOperator, ColumnRef, Expr, FunctionCall, Literal, UnaryOperator};
use super::types::{Row, Schema};
use sochdb_core::Result;

/// Evaluate an expression against a row, producing a scalar value.
pub fn eval_expr(expr: &Expr, row: &Row, schema: &Schema) -> Result<SochValue> {
    match expr {
        Expr::Literal(lit) => Ok(eval_literal(lit)),

        Expr::Column(col_ref) => eval_column(col_ref, row, schema),

        Expr::BinaryOp { left, op, right } => {
            let lv = eval_expr(left, row, schema)?;
            // Short-circuit AND/OR
            match op {
                BinaryOperator::And => {
                    if !value_is_truthy(&lv) {
                        return Ok(SochValue::Bool(false));
                    }
                    let rv = eval_expr(right, row, schema)?;
                    return Ok(SochValue::Bool(value_is_truthy(&rv)));
                }
                BinaryOperator::Or => {
                    if value_is_truthy(&lv) {
                        return Ok(SochValue::Bool(true));
                    }
                    let rv = eval_expr(right, row, schema)?;
                    return Ok(SochValue::Bool(value_is_truthy(&rv)));
                }
                _ => {}
            }
            let rv = eval_expr(right, row, schema)?;
            eval_binary_op(&lv, *op, &rv)
        }

        Expr::UnaryOp { op, expr: inner } => {
            let v = eval_expr(inner, row, schema)?;
            eval_unary_op(*op, &v)
        }

        Expr::Function(func) => eval_function(func, row, schema),

        Expr::IsNull { expr: inner, negated } => {
            let v = eval_expr(inner, row, schema)?;
            let is_null = matches!(v, SochValue::Null);
            Ok(SochValue::Bool(if *negated { !is_null } else { is_null }))
        }

        Expr::Between { expr: inner, low, high, negated } => {
            let v = eval_expr(inner, row, schema)?;
            let lo = eval_expr(low, row, schema)?;
            let hi = eval_expr(high, row, schema)?;
            let in_range = compare_values(&v, &lo) != Some(std::cmp::Ordering::Less)
                && compare_values(&v, &hi) != Some(std::cmp::Ordering::Greater);
            Ok(SochValue::Bool(if *negated { !in_range } else { in_range }))
        }

        Expr::InList { expr: inner, list, negated } => {
            let v = eval_expr(inner, row, schema)?;
            let mut found = false;
            for item in list {
                let iv = eval_expr(item, row, schema)?;
                if compare_values(&v, &iv) == Some(std::cmp::Ordering::Equal) {
                    found = true;
                    break;
                }
            }
            Ok(SochValue::Bool(if *negated { !found } else { found }))
        }

        Expr::Like { expr: inner, pattern, negated, .. } => {
            let v = eval_expr(inner, row, schema)?;
            let p = eval_expr(pattern, row, schema)?;
            let matched = match (&v, &p) {
                (SochValue::Text(s), SochValue::Text(pat)) => like_match(s, pat),
                _ => false,
            };
            Ok(SochValue::Bool(if *negated { !matched } else { matched }))
        }

        Expr::Case { operand, conditions, else_result } => {
            eval_case(operand.as_deref(), conditions, else_result.as_deref(), row, schema)
        }

        Expr::Cast { expr: inner, data_type: _ } => {
            // Simplified: just evaluate the inner expression
            // Full type coercion can be added later
            eval_expr(inner, row, schema)
        }

        Expr::Array(elements) => {
            let mut vals = Vec::with_capacity(elements.len());
            for e in elements {
                vals.push(eval_expr(e, row, schema)?);
            }
            Ok(SochValue::Array(vals))
        }

        Expr::Tuple(elements) => {
            let mut vals = Vec::with_capacity(elements.len());
            for e in elements {
                vals.push(eval_expr(e, row, schema)?);
            }
            Ok(SochValue::Array(vals))
        }

        // Placeholders should have been resolved before evaluation
        Expr::Placeholder(_) => Err(sochdb_core::SochDBError::Internal(
            "Unresolved placeholder in expression".into(),
        )),

        // Vector literal
        Expr::Vector(floats) => {
            Ok(SochValue::Array(
                floats.iter().map(|f| SochValue::Float(*f as f64)).collect(),
            ))
        }

        // Not yet supported in executor
        Expr::Subquery(_)
        | Expr::Exists(_)
        | Expr::InSubquery { .. }
        | Expr::VectorSearch { .. }
        | Expr::JsonAccess { .. }
        | Expr::ContextWindow { .. }
        | Expr::Subscript { .. }
        | Expr::RecordId { .. } => Err(sochdb_core::SochDBError::Internal(
            format!("Expression type not yet supported in executor: {:?}", std::mem::discriminant(expr)),
        )),
    }
}

/// Evaluate a predicate expression, returning a boolean.
pub fn eval_predicate(expr: &Expr, row: &Row, schema: &Schema) -> Result<bool> {
    let v = eval_expr(expr, row, schema)?;
    Ok(value_is_truthy(&v))
}

// ============================================================================
// Internal helpers
// ============================================================================

fn eval_literal(lit: &Literal) -> SochValue {
    match lit {
        Literal::Null => SochValue::Null,
        Literal::Boolean(b) => SochValue::Bool(*b),
        Literal::Integer(i) => SochValue::Int(*i),
        Literal::Float(f) => SochValue::Float(*f),
        Literal::String(s) => SochValue::Text(s.clone()),
        Literal::Blob(b) => SochValue::Binary(b.clone()),
    }
}

fn eval_column(col: &ColumnRef, row: &Row, schema: &Schema) -> Result<SochValue> {
    let idx = schema
        .index_of_qualified(col.table.as_deref(), &col.column)
        .ok_or_else(|| {
            sochdb_core::SochDBError::Internal(format!(
                "Column '{}' not found in schema {:?}",
                col.column,
                schema.column_names()
            ))
        })?;
    Ok(row.get(idx).cloned().unwrap_or(SochValue::Null))
}

fn eval_binary_op(lv: &SochValue, op: BinaryOperator, rv: &SochValue) -> Result<SochValue> {
    // NULL propagation for most ops
    if matches!(lv, SochValue::Null) || matches!(rv, SochValue::Null) {
        match op {
            BinaryOperator::Eq | BinaryOperator::Ne | BinaryOperator::Lt
            | BinaryOperator::Le | BinaryOperator::Gt | BinaryOperator::Ge => {
                // NULL compared to anything is NULL (SQL three-valued logic)
                return Ok(SochValue::Null);
            }
            _ => return Ok(SochValue::Null),
        }
    }

    match op {
        // Comparison operators
        BinaryOperator::Eq => Ok(SochValue::Bool(
            compare_values(lv, rv) == Some(std::cmp::Ordering::Equal),
        )),
        BinaryOperator::Ne => Ok(SochValue::Bool(
            compare_values(lv, rv) != Some(std::cmp::Ordering::Equal),
        )),
        BinaryOperator::Lt => Ok(SochValue::Bool(
            compare_values(lv, rv) == Some(std::cmp::Ordering::Less),
        )),
        BinaryOperator::Le => Ok(SochValue::Bool(
            compare_values(lv, rv) != Some(std::cmp::Ordering::Greater),
        )),
        BinaryOperator::Gt => Ok(SochValue::Bool(
            compare_values(lv, rv) == Some(std::cmp::Ordering::Greater),
        )),
        BinaryOperator::Ge => Ok(SochValue::Bool(
            compare_values(lv, rv) != Some(std::cmp::Ordering::Less),
        )),

        // Arithmetic
        BinaryOperator::Plus => eval_arithmetic(lv, rv, |a, b| a + b, |a, b| a + b),
        BinaryOperator::Minus => eval_arithmetic(lv, rv, |a, b| a - b, |a, b| a - b),
        BinaryOperator::Multiply => eval_arithmetic(lv, rv, |a, b| a * b, |a, b| a * b),
        BinaryOperator::Divide => {
            // Division by zero check
            match rv {
                SochValue::Int(0) | SochValue::UInt(0) => {
                    return Err(sochdb_core::SochDBError::Internal("Division by zero".into()));
                }
                SochValue::Float(f) if *f == 0.0 => {
                    return Err(sochdb_core::SochDBError::Internal("Division by zero".into()));
                }
                _ => {}
            }
            eval_arithmetic(lv, rv, |a, b| a / b, |a, b| a / b)
        }
        BinaryOperator::Modulo => {
            match rv {
                SochValue::Int(0) | SochValue::UInt(0) => {
                    return Err(sochdb_core::SochDBError::Internal("Division by zero".into()));
                }
                _ => {}
            }
            eval_arithmetic(lv, rv, |a, b| a % b, |a, b| a % b)
        }

        // String concat
        BinaryOperator::Concat => {
            let ls = value_to_string(lv);
            let rs = value_to_string(rv);
            Ok(SochValue::Text(format!("{}{}", ls, rs)))
        }

        // LIKE handled separately above in eval_expr
        BinaryOperator::Like => {
            match (lv, rv) {
                (SochValue::Text(s), SochValue::Text(p)) => {
                    Ok(SochValue::Bool(like_match(s, p)))
                }
                _ => Ok(SochValue::Bool(false)),
            }
        }

        // Bitwise ops
        BinaryOperator::BitAnd => eval_bitwise(lv, rv, |a, b| a & b),
        BinaryOperator::BitOr => eval_bitwise(lv, rv, |a, b| a | b),
        BinaryOperator::BitXor => eval_bitwise(lv, rv, |a, b| a ^ b),
        BinaryOperator::LeftShift => eval_bitwise(lv, rv, |a, b| a << b),
        BinaryOperator::RightShift => eval_bitwise(lv, rv, |a, b| a >> b),

        // AND/OR handled above with short-circuit
        BinaryOperator::And | BinaryOperator::Or => unreachable!(),

        // Graph traversal — requires graph execution engine (not yet implemented)
        BinaryOperator::GraphRight | BinaryOperator::GraphLeft | BinaryOperator::GraphBi => {
            Err(sochdb_core::SochDBError::Internal(
                "Graph traversal operators (-> <- <->) not yet supported in scalar evaluation".into(),
            ))
        }
    }
}

fn eval_unary_op(op: UnaryOperator, v: &SochValue) -> Result<SochValue> {
    match op {
        UnaryOperator::Minus => match v {
            SochValue::Int(i) => Ok(SochValue::Int(-i)),
            SochValue::Float(f) => Ok(SochValue::Float(-f)),
            SochValue::Null => Ok(SochValue::Null),
            _ => Err(sochdb_core::SochDBError::Internal("Cannot negate non-numeric value".into())),
        },
        UnaryOperator::Plus => Ok(v.clone()),
        UnaryOperator::Not => Ok(SochValue::Bool(!value_is_truthy(v))),
        UnaryOperator::BitNot => match v {
            SochValue::Int(i) => Ok(SochValue::Int(!i)),
            SochValue::Null => Ok(SochValue::Null),
            _ => Err(sochdb_core::SochDBError::Internal("Cannot bitwise-NOT non-integer".into())),
        },
    }
}

fn eval_function(func: &FunctionCall, row: &Row, schema: &Schema) -> Result<SochValue> {
    let name = func.name.name().to_uppercase();

    // Evaluate arguments
    let args: Vec<SochValue> = func
        .args
        .iter()
        .map(|a| eval_expr(a, row, schema))
        .collect::<Result<Vec<_>>>()?;

    match name.as_str() {
        "COALESCE" => {
            for a in &args {
                if !matches!(a, SochValue::Null) {
                    return Ok(a.clone());
                }
            }
            Ok(SochValue::Null)
        }
        "NULLIF" => {
            if args.len() == 2 && compare_values(&args[0], &args[1]) == Some(std::cmp::Ordering::Equal) {
                Ok(SochValue::Null)
            } else {
                Ok(args.into_iter().next().unwrap_or(SochValue::Null))
            }
        }
        "ABS" => match args.first() {
            Some(SochValue::Int(i)) => Ok(SochValue::Int(i.abs())),
            Some(SochValue::Float(f)) => Ok(SochValue::Float(f.abs())),
            _ => Ok(SochValue::Null),
        },
        "LENGTH" | "LEN" => match args.first() {
            Some(SochValue::Text(s)) => Ok(SochValue::Int(s.len() as i64)),
            Some(SochValue::Binary(b)) => Ok(SochValue::Int(b.len() as i64)),
            _ => Ok(SochValue::Null),
        },
        "UPPER" => match args.first() {
            Some(SochValue::Text(s)) => Ok(SochValue::Text(s.to_uppercase())),
            _ => Ok(SochValue::Null),
        },
        "LOWER" => match args.first() {
            Some(SochValue::Text(s)) => Ok(SochValue::Text(s.to_lowercase())),
            _ => Ok(SochValue::Null),
        },
        "TRIM" => match args.first() {
            Some(SochValue::Text(s)) => Ok(SochValue::Text(s.trim().to_string())),
            _ => Ok(SochValue::Null),
        },
        "SUBSTR" | "SUBSTRING" => {
            match (args.get(0), args.get(1), args.get(2)) {
                (Some(SochValue::Text(s)), Some(SochValue::Int(start)), len) => {
                    let start_idx = (*start as usize).saturating_sub(1); // SQL is 1-indexed
                    let slice = if let Some(SochValue::Int(l)) = len {
                        let end = start_idx + (*l as usize);
                        &s[start_idx..end.min(s.len())]
                    } else {
                        &s[start_idx..]
                    };
                    Ok(SochValue::Text(slice.to_string()))
                }
                _ => Ok(SochValue::Null),
            }
        }
        "CONCAT" => {
            let s: String = args.iter().map(value_to_string).collect::<Vec<_>>().join("");
            Ok(SochValue::Text(s))
        }
        "REPLACE" => {
            match (args.get(0), args.get(1), args.get(2)) {
                (Some(SochValue::Text(s)), Some(SochValue::Text(from)), Some(SochValue::Text(to))) => {
                    Ok(SochValue::Text(s.replace(from.as_str(), to.as_str())))
                }
                _ => Ok(SochValue::Null),
            }
        }
        "ROUND" => {
            match (args.get(0), args.get(1)) {
                (Some(SochValue::Float(f)), Some(SochValue::Int(digits))) => {
                    let factor = 10f64.powi(*digits as i32);
                    Ok(SochValue::Float((f * factor).round() / factor))
                }
                (Some(SochValue::Float(f)), None) => Ok(SochValue::Float(f.round())),
                (Some(SochValue::Int(i)), _) => Ok(SochValue::Int(*i)),
                _ => Ok(SochValue::Null),
            }
        }
        "FLOOR" => match args.first() {
            Some(SochValue::Float(f)) => Ok(SochValue::Float(f.floor())),
            Some(v @ SochValue::Int(_)) => Ok(v.clone()),
            _ => Ok(SochValue::Null),
        },
        "CEIL" | "CEILING" => match args.first() {
            Some(SochValue::Float(f)) => Ok(SochValue::Float(f.ceil())),
            Some(v @ SochValue::Int(_)) => Ok(v.clone()),
            _ => Ok(SochValue::Null),
        },
        "GREATEST" => {
            let mut best: Option<SochValue> = None;
            for a in &args {
                if matches!(a, SochValue::Null) { continue; }
                best = Some(match &best {
                    None => a.clone(),
                    Some(b) => {
                        if compare_values(a, b) == Some(std::cmp::Ordering::Greater) {
                            a.clone()
                        } else {
                            b.clone()
                        }
                    }
                });
            }
            Ok(best.unwrap_or(SochValue::Null))
        }
        "LEAST" => {
            let mut best: Option<SochValue> = None;
            for a in &args {
                if matches!(a, SochValue::Null) { continue; }
                best = Some(match &best {
                    None => a.clone(),
                    Some(b) => {
                        if compare_values(a, b) == Some(std::cmp::Ordering::Less) {
                            a.clone()
                        } else {
                            b.clone()
                        }
                    }
                });
            }
            Ok(best.unwrap_or(SochValue::Null))
        }
        // Aggregate function names — evaluated at aggregate operator level, not per-row.
        // If we get here it's because the query references an aggregate in a non-aggregate context.
        "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "COUNT_DISTINCT" => {
            Err(sochdb_core::SochDBError::Internal(format!(
                "Aggregate function {}() cannot be used outside of GROUP BY context",
                name,
            )))
        }
        _ => Err(sochdb_core::SochDBError::Internal(format!(
            "Unknown function: {}",
            name,
        ))),
    }
}

fn eval_case(
    operand: Option<&Expr>,
    conditions: &[(Expr, Expr)],
    else_result: Option<&Expr>,
    row: &Row,
    schema: &Schema,
) -> Result<SochValue> {
    if let Some(op) = operand {
        // Simple CASE: CASE operand WHEN val1 THEN result1 ...
        let op_val = eval_expr(op, row, schema)?;
        for (when_expr, then_expr) in conditions {
            let when_val = eval_expr(when_expr, row, schema)?;
            if compare_values(&op_val, &when_val) == Some(std::cmp::Ordering::Equal) {
                return eval_expr(then_expr, row, schema);
            }
        }
    } else {
        // Searched CASE: CASE WHEN cond1 THEN result1 ...
        for (when_expr, then_expr) in conditions {
            if eval_predicate(when_expr, row, schema)? {
                return eval_expr(then_expr, row, schema);
            }
        }
    }
    match else_result {
        Some(e) => eval_expr(e, row, schema),
        None => Ok(SochValue::Null),
    }
}

// ============================================================================
// Comparison and type coercion utilities
// ============================================================================

/// Compare two SochValues, returning ordering if comparable.
pub fn compare_values(a: &SochValue, b: &SochValue) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;

    match (a, b) {
        (SochValue::Null, SochValue::Null) => Some(Ordering::Equal),
        (SochValue::Null, _) | (_, SochValue::Null) => None,
        (SochValue::Bool(a), SochValue::Bool(b)) => a.partial_cmp(b),
        (SochValue::Int(a), SochValue::Int(b)) => a.partial_cmp(b),
        (SochValue::UInt(a), SochValue::UInt(b)) => a.partial_cmp(b),
        (SochValue::Float(a), SochValue::Float(b)) => a.partial_cmp(b),
        (SochValue::Text(a), SochValue::Text(b)) => a.partial_cmp(b),

        // Cross-type numeric comparisons
        (SochValue::Int(a), SochValue::Float(b)) => (*a as f64).partial_cmp(b),
        (SochValue::Float(a), SochValue::Int(b)) => a.partial_cmp(&(*b as f64)),
        (SochValue::Int(a), SochValue::UInt(b)) => (*a as i128).partial_cmp(&(*b as i128)),
        (SochValue::UInt(a), SochValue::Int(b)) => (*a as i128).partial_cmp(&(*b as i128)),
        (SochValue::UInt(a), SochValue::Float(b)) => (*a as f64).partial_cmp(b),
        (SochValue::Float(a), SochValue::UInt(b)) => a.partial_cmp(&(*b as f64)),

        _ => None,
    }
}

/// SQL LIKE pattern matching (% and _ wildcards).
fn like_match(s: &str, pattern: &str) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    let p_chars: Vec<char> = pattern.chars().collect();
    like_match_impl(&s_chars, &p_chars, 0, 0)
}

fn like_match_impl(s: &[char], p: &[char], si: usize, pi: usize) -> bool {
    if pi == p.len() {
        return si == s.len();
    }
    match p[pi] {
        '%' => {
            // % matches zero or more characters
            for i in si..=s.len() {
                if like_match_impl(s, p, i, pi + 1) {
                    return true;
                }
            }
            false
        }
        '_' => {
            // _ matches exactly one character
            si < s.len() && like_match_impl(s, p, si + 1, pi + 1)
        }
        c => {
            si < s.len()
                && (s[si] == c || s[si].to_lowercase().eq(c.to_lowercase()))
                && like_match_impl(s, p, si + 1, pi + 1)
        }
    }
}

/// Check if a value is truthy (for predicate evaluation).
pub fn value_is_truthy(v: &SochValue) -> bool {
    match v {
        SochValue::Bool(b) => *b,
        SochValue::Int(i) => *i != 0,
        SochValue::UInt(u) => *u != 0,
        SochValue::Float(f) => *f != 0.0,
        SochValue::Text(s) => !s.is_empty(),
        SochValue::Null => false,
        _ => true,
    }
}

/// Convert a SochValue to a display string.
fn value_to_string(v: &SochValue) -> String {
    match v {
        SochValue::Null => "NULL".to_string(),
        SochValue::Bool(b) => b.to_string(),
        SochValue::Int(i) => i.to_string(),
        SochValue::UInt(u) => u.to_string(),
        SochValue::Float(f) => f.to_string(),
        SochValue::Text(s) => s.clone(),
        SochValue::Binary(b) => format!("0x{}", hex::encode(b)),
        SochValue::Array(a) => format!("[{}]", a.iter().map(value_to_string).collect::<Vec<_>>().join(",")),
    }
}

fn eval_arithmetic<F, G>(lv: &SochValue, rv: &SochValue, int_op: F, float_op: G) -> Result<SochValue>
where
    F: Fn(i64, i64) -> i64,
    G: Fn(f64, f64) -> f64,
{
    match (lv, rv) {
        (SochValue::Int(a), SochValue::Int(b)) => Ok(SochValue::Int(int_op(*a, *b))),
        (SochValue::Float(a), SochValue::Float(b)) => Ok(SochValue::Float(float_op(*a, *b))),
        (SochValue::Int(a), SochValue::Float(b)) => Ok(SochValue::Float(float_op(*a as f64, *b))),
        (SochValue::Float(a), SochValue::Int(b)) => Ok(SochValue::Float(float_op(*a, *b as f64))),
        (SochValue::UInt(a), SochValue::UInt(b)) => Ok(SochValue::Int(int_op(*a as i64, *b as i64))),
        (SochValue::Int(a), SochValue::UInt(b)) => Ok(SochValue::Int(int_op(*a, *b as i64))),
        (SochValue::UInt(a), SochValue::Int(b)) => Ok(SochValue::Int(int_op(*a as i64, *b))),
        (SochValue::UInt(a), SochValue::Float(b)) => Ok(SochValue::Float(float_op(*a as f64, *b))),
        (SochValue::Float(a), SochValue::UInt(b)) => Ok(SochValue::Float(float_op(*a, *b as f64))),
        _ => Err(sochdb_core::SochDBError::Internal(format!(
            "Cannot perform arithmetic on {:?} and {:?}",
            std::mem::discriminant(lv),
            std::mem::discriminant(rv),
        ))),
    }
}

fn eval_bitwise<F>(lv: &SochValue, rv: &SochValue, op: F) -> Result<SochValue>
where
    F: Fn(i64, i64) -> i64,
{
    match (lv, rv) {
        (SochValue::Int(a), SochValue::Int(b)) => Ok(SochValue::Int(op(*a, *b))),
        (SochValue::UInt(a), SochValue::UInt(b)) => Ok(SochValue::Int(op(*a as i64, *b as i64))),
        _ => Err(sochdb_core::SochDBError::Internal("Bitwise ops require integer operands".into())),
    }
}

// Inline hex encoding to avoid extra dependency
mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

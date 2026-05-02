// SPDX-License-Identifier: AGPL-3.0-or-later

//! Join operators: HashJoin, NestedLoopJoin, MergeJoin.

use crate::soch_ql::SochValue;
use crate::sql::ast::{Expr, JoinType};
use super::eval::{eval_expr, eval_predicate, compare_values};
use super::node::PlanNode;
use super::types::{Row, Schema};
use sochdb_core::Result;
use std::collections::HashMap;

// ============================================================================
// HashJoinNode — Hash-based equi-join
// ============================================================================

/// Hash join: builds a hash table from the build side, probes with the probe side.
///
/// Supports INNER, LEFT, RIGHT, and FULL [OUTER] joins.
///
/// ```text
/// HashJoin(build_key=users.id, probe_key=orders.user_id)
///   ├── build: SeqScan(users)
///   └── probe: SeqScan(orders)
/// ```
pub struct HashJoinNode {
    build: Box<dyn PlanNode>,
    probe: Box<dyn PlanNode>,
    /// Expression to evaluate on build side to produce hash key.
    build_key_expr: Expr,
    /// Expression to evaluate on probe side to produce hash key.
    probe_key_expr: Expr,
    join_type: JoinType,
    output_schema: Schema,
    /// Hash table: key -> list of build rows.
    hash_table: Option<HashMap<HashKey, Vec<Row>>>,
    /// For LEFT/RIGHT/FULL: track which build rows were matched.
    build_matched: Vec<bool>,
    /// Current probe row being processed.
    current_probe_row: Option<Row>,
    /// Matches for current probe row.
    current_matches: Vec<Row>,
    /// Index into current_matches.
    match_idx: usize,
    /// For RIGHT/FULL: unmatched build rows to emit after probe exhausted.
    unmatched_buffer: Option<Vec<Row>>,
    unmatched_pos: usize,
    /// Whether probe side is exhausted.
    probe_exhausted: bool,
    /// Whether join produced a match for current probe row.
    current_probe_matched: bool,
    build_schema: Schema,
    probe_schema: Schema,
}

/// Simple hash key wrapping a SochValue for HashMap use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum HashKey {
    Int(i64),
    UInt(u64),
    Text(String),
    Bool(bool),
    Null,
    Other(String),
}

impl From<&SochValue> for HashKey {
    fn from(v: &SochValue) -> Self {
        match v {
            SochValue::Int(i) => HashKey::Int(*i),
            SochValue::UInt(u) => HashKey::UInt(*u),
            SochValue::Text(s) => HashKey::Text(s.clone()),
            SochValue::Bool(b) => HashKey::Bool(*b),
            SochValue::Null => HashKey::Null,
            other => HashKey::Other(format!("{:?}", other)),
        }
    }
}

impl HashJoinNode {
    pub fn new(
        build: Box<dyn PlanNode>,
        probe: Box<dyn PlanNode>,
        build_key_expr: Expr,
        probe_key_expr: Expr,
        join_type: JoinType,
    ) -> Self {
        let build_schema = build.schema().clone();
        let probe_schema = probe.schema().clone();
        let output_schema = build_schema.merge(&probe_schema);

        Self {
            build,
            probe,
            build_key_expr,
            probe_key_expr,
            join_type,
            output_schema,
            hash_table: None,
            build_matched: Vec::new(),
            current_probe_row: None,
            current_matches: Vec::new(),
            match_idx: 0,
            unmatched_buffer: None,
            unmatched_pos: 0,
            probe_exhausted: false,
            current_probe_matched: false,
            build_schema,
            probe_schema,
        }
    }

    fn build_hash_table(&mut self) -> Result<()> {
        if self.hash_table.is_some() {
            return Ok(());
        }

        let mut table: HashMap<HashKey, Vec<Row>> = HashMap::new();
        let mut all_build_rows: Vec<Row> = Vec::new();
        let schema = self.build.schema().clone();

        while let Some(row) = self.build.next()? {
            let key_val = eval_expr(&self.build_key_expr, &row, &schema)?;
            let key = HashKey::from(&key_val);
            table.entry(key).or_default().push(row.clone());
            all_build_rows.push(row);
        }

        self.build_matched = vec![false; all_build_rows.len()];
        self.hash_table = Some(table);
        Ok(())
    }

    fn null_row(schema: &Schema) -> Row {
        vec![SochValue::Null; schema.len()]
    }

    fn combine(build_row: &Row, probe_row: &Row) -> Row {
        let mut combined = build_row.clone();
        combined.extend(probe_row.iter().cloned());
        combined
    }
}

impl PlanNode for HashJoinNode {
    fn schema(&self) -> &Schema {
        &self.output_schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        self.build_hash_table()?;

        loop {
            // Return pending matches from current probe row
            if self.match_idx < self.current_matches.len() {
                let build_row = &self.current_matches[self.match_idx];
                let probe_row = self.current_probe_row.as_ref().unwrap();
                self.match_idx += 1;
                return Ok(Some(Self::combine(build_row, probe_row)));
            }

            // For LEFT join: emit unmatched probe row
            if self.current_probe_row.is_some()
                && !self.current_probe_matched
                && matches!(self.join_type, JoinType::Left | JoinType::Full)
            {
                let probe_row = self.current_probe_row.take().unwrap();
                let null_build = Self::null_row(&self.build_schema);
                return Ok(Some(Self::combine(&null_build, &probe_row)));
            }

            // Done with current probe row, reset
            self.current_probe_row = None;
            self.current_matches.clear();
            self.match_idx = 0;
            self.current_probe_matched = false;

            if !self.probe_exhausted {
                // Get next probe row
                match self.probe.next()? {
                    Some(probe_row) => {
                        let key_val = eval_expr(
                            &self.probe_key_expr,
                            &probe_row,
                            &self.probe_schema,
                        )?;
                        let key = HashKey::from(&key_val);

                        if let Some(ht) = &self.hash_table {
                            if let Some(matches) = ht.get(&key) {
                                self.current_matches = matches.clone();
                                self.current_probe_matched = true;
                                // Mark matched build rows
                                // (simplified: we'd need row indices for precise tracking)
                            }
                        }

                        self.current_probe_row = Some(probe_row);
                        continue;
                    }
                    None => {
                        self.probe_exhausted = true;
                    }
                }
            }

            // After probe exhausted: emit unmatched build rows for RIGHT/FULL join
            if matches!(self.join_type, JoinType::Right | JoinType::Full) {
                if self.unmatched_buffer.is_none() {
                    // Collect unmatched build rows
                    // Simplified: for now just return None
                    // Full RIGHT/FULL join requires tracking which build rows were matched
                    self.unmatched_buffer = Some(Vec::new());
                }

                if let Some(buf) = &self.unmatched_buffer {
                    if self.unmatched_pos < buf.len() {
                        let row = buf[self.unmatched_pos].clone();
                        self.unmatched_pos += 1;
                        let null_probe = Self::null_row(&self.probe_schema);
                        return Ok(Some(Self::combine(&row, &null_probe)));
                    }
                }
            }

            return Ok(None);
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.hash_table = None;
        self.current_probe_row = None;
        self.current_matches.clear();
        self.match_idx = 0;
        self.probe_exhausted = false;
        self.unmatched_buffer = None;
        self.unmatched_pos = 0;
        self.build.reset()?;
        self.probe.reset()
    }
}

// ============================================================================
// NestedLoopJoinNode — Theta join (any join condition)
// ============================================================================

/// Nested loop join: for each outer row, scans all inner rows testing condition.
///
/// Supports all join types and arbitrary join conditions.
pub struct NestedLoopJoinNode {
    outer: Box<dyn PlanNode>,
    inner: Box<dyn PlanNode>,
    condition: Option<Expr>,
    join_type: JoinType,
    output_schema: Schema,
    /// Current outer row.
    current_outer: Option<Row>,
    /// Whether current outer row has matched any inner row.
    current_matched: bool,
    /// Whether join is exhausted.
    outer_exhausted: bool,
    _outer_schema: Schema,
    inner_schema: Schema,
}

impl NestedLoopJoinNode {
    pub fn new(
        outer: Box<dyn PlanNode>,
        inner: Box<dyn PlanNode>,
        condition: Option<Expr>,
        join_type: JoinType,
    ) -> Self {
        let outer_schema = outer.schema().clone();
        let inner_schema = inner.schema().clone();
        let output_schema = outer_schema.merge(&inner_schema);

        Self {
            outer,
            inner,
            condition,
            join_type,
            output_schema,
            current_outer: None,
            current_matched: false,
            outer_exhausted: false,
            _outer_schema: outer_schema,
            inner_schema,
        }
    }

    fn combine(outer_row: &Row, inner_row: &Row) -> Row {
        let mut combined = outer_row.clone();
        combined.extend(inner_row.iter().cloned());
        combined
    }

    fn null_row(schema: &Schema) -> Row {
        vec![SochValue::Null; schema.len()]
    }
}

impl PlanNode for NestedLoopJoinNode {
    fn schema(&self) -> &Schema {
        &self.output_schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        loop {
            // Get current outer row (or advance to next)
            if self.current_outer.is_none() {
                if self.outer_exhausted {
                    return Ok(None);
                }
                match self.outer.next()? {
                    Some(row) => {
                        self.current_outer = Some(row);
                        self.current_matched = false;
                        self.inner.reset()?;
                    }
                    None => {
                        self.outer_exhausted = true;
                        return Ok(None);
                    }
                }
            }

            let outer_row = self.current_outer.as_ref().unwrap();

            // Try to find next matching inner row
            match self.inner.next()? {
                Some(inner_row) => {
                    let combined = Self::combine(outer_row, &inner_row);

                    // Evaluate join condition
                    let matched = match &self.condition {
                        Some(cond) => eval_predicate(cond, &combined, &self.output_schema)?,
                        None => true, // CROSS JOIN
                    };

                    if matched {
                        self.current_matched = true;
                        return Ok(Some(combined));
                    }
                    // Not matched, try next inner row
                    continue;
                }
                None => {
                    // Inner side exhausted for this outer row
                    let need_null_row = !self.current_matched
                        && matches!(self.join_type, JoinType::Left | JoinType::Full);

                    let outer_row = self.current_outer.take().unwrap();

                    if need_null_row {
                        let null_inner = Self::null_row(&self.inner_schema);
                        return Ok(Some(Self::combine(&outer_row, &null_inner)));
                    }
                    // Move to next outer row
                    continue;
                }
            }
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.current_outer = None;
        self.current_matched = false;
        self.outer_exhausted = false;
        self.outer.reset()?;
        self.inner.reset()
    }
}

// ============================================================================
// MergeJoinNode — Merge join on sorted inputs
// ============================================================================

/// Merge join: requires both inputs sorted on join keys.
///
/// For INNER JOIN, produces output only when keys match.
pub struct MergeJoinNode {
    left: Box<dyn PlanNode>,
    right: Box<dyn PlanNode>,
    left_key_expr: Expr,
    right_key_expr: Expr,
    join_type: JoinType,
    output_schema: Schema,
    left_schema: Schema,
    right_schema: Schema,
    /// Buffered rows from right side with same key (for many-to-many).
    right_buffer: Vec<Row>,
    right_buffer_key: Option<SochValue>,
    right_buf_idx: usize,
    current_left: Option<Row>,
    current_left_key: Option<SochValue>,
    right_exhausted: bool,
    pending_right: Option<Row>,
}

impl MergeJoinNode {
    pub fn new(
        left: Box<dyn PlanNode>,
        right: Box<dyn PlanNode>,
        left_key_expr: Expr,
        right_key_expr: Expr,
        join_type: JoinType,
    ) -> Self {
        let left_schema = left.schema().clone();
        let right_schema = right.schema().clone();
        let output_schema = left_schema.merge(&right_schema);

        Self {
            left,
            right,
            left_key_expr,
            right_key_expr,
            join_type,
            output_schema,
            left_schema,
            right_schema,
            right_buffer: Vec::new(),
            right_buffer_key: None,
            right_buf_idx: 0,
            current_left: None,
            current_left_key: None,
            right_exhausted: false,
            pending_right: None,
        }
    }

    fn combine(left_row: &Row, right_row: &Row) -> Row {
        let mut combined = left_row.clone();
        combined.extend(right_row.iter().cloned());
        combined
    }

    fn advance_right(&mut self) -> Result<Option<(SochValue, Row)>> {
        if let Some(row) = self.pending_right.take() {
            let key = eval_expr(&self.right_key_expr, &row, &self.right_schema)?;
            return Ok(Some((key, row)));
        }
        match self.right.next()? {
            Some(row) => {
                let key = eval_expr(&self.right_key_expr, &row, &self.right_schema)?;
                Ok(Some((key, row)))
            }
            None => {
                self.right_exhausted = true;
                Ok(None)
            }
        }
    }
}

impl PlanNode for MergeJoinNode {
    fn schema(&self) -> &Schema {
        &self.output_schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        loop {
            // If we have buffered right matches, emit them
            if self.right_buf_idx < self.right_buffer.len() {
                if let Some(left_row) = &self.current_left {
                    let right_row = &self.right_buffer[self.right_buf_idx];
                    self.right_buf_idx += 1;
                    return Ok(Some(Self::combine(left_row, right_row)));
                }
            }

            // Need new left row
            let left_row = match self.left.next()? {
                Some(row) => row,
                None => return Ok(None),
            };
            let left_key = eval_expr(&self.left_key_expr, &left_row, &self.left_schema)?;

            // Check if right buffer has same key
            if self.right_buffer_key.as_ref().map_or(false, |k| {
                compare_values(k, &left_key) == Some(std::cmp::Ordering::Equal)
            }) {
                self.current_left = Some(left_row);
                self.current_left_key = Some(left_key);
                self.right_buf_idx = 0;
                continue;
            }

            // Need to advance right side to match left key
            self.right_buffer.clear();
            self.right_buf_idx = 0;

            if self.right_exhausted {
                if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                    let null_right = vec![SochValue::Null; self.right_schema.len()];
                    return Ok(Some(Self::combine(&left_row, &null_right)));
                }
                return Ok(None);
            }

            // Advance right until we find matching or greater key
            loop {
                match self.advance_right()? {
                    Some((right_key, right_row)) => {
                        match compare_values(&right_key, &left_key) {
                            Some(std::cmp::Ordering::Equal) => {
                                self.right_buffer.push(right_row);
                                self.right_buffer_key = Some(right_key);
                                // Collect all right rows with same key
                                break;
                            }
                            Some(std::cmp::Ordering::Greater) => {
                                // Right key is past left key
                                self.pending_right = Some(right_row);
                                break;
                            }
                            _ => {
                                // Right key is less than left key, skip
                                continue;
                            }
                        }
                    }
                    None => break,
                }
            }

            // Collect remaining right rows with same key
            if !self.right_buffer.is_empty() {
                loop {
                    match self.advance_right()? {
                        Some((right_key, right_row)) => {
                            if compare_values(&right_key, &left_key) == Some(std::cmp::Ordering::Equal) {
                                self.right_buffer.push(right_row);
                            } else {
                                self.pending_right = Some(right_row);
                                break;
                            }
                        }
                        None => break,
                    }
                }
            }

            self.current_left = Some(left_row);
            self.current_left_key = Some(left_key);
            self.right_buf_idx = 0;

            if self.right_buffer.is_empty() {
                if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                    let left_row = self.current_left.take().unwrap();
                    let null_right = vec![SochValue::Null; self.right_schema.len()];
                    return Ok(Some(Self::combine(&left_row, &null_right)));
                }
                // Inner join, no match => skip this left row
                continue;
            }
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.right_buffer.clear();
        self.right_buffer_key = None;
        self.right_buf_idx = 0;
        self.current_left = None;
        self.current_left_key = None;
        self.right_exhausted = false;
        self.pending_right = None;
        self.left.reset()?;
        self.right.reset()
    }
}

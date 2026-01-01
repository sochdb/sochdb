// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Filter/Projection Pushdown
//!
//! This module provides a plugin API for user-defined filter predicates
//! that can be evaluated early in the search pipeline.
//!
//! # Problem
//!
//! Late filtering wastes work:
//! - Compute distances for filtered-out vectors
//! - Rerank candidates that fail predicates
//! - Memory bandwidth wasted on rejected data
//!
//! # Solution
//!
//! Pushdown evaluation:
//! 1. Define filter predicates via plugin API
//! 2. Evaluate predicates early in pipeline (before distance computation)
//! 3. Support compiled predicates (WASM) for performance
//! 4. Projection to reduce data transfer
//!
//! # Predicate Evaluation Order
//!
//! 1. Selectivity estimation
//! 2. Cost-based ordering
//! 3. Short-circuit evaluation
//!
//! # Plugin Types
//!
//! - Native Rust closures (fastest)
//! - WASM modules (sandboxed, portable)
//! - Expression trees (interpretable, optimizable)

use std::collections::HashMap;
use std::sync::Arc;

/// Result type for filter operations.
pub type FilterResult<T> = Result<T, FilterError>;

/// Filter operation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterError {
    /// Invalid predicate expression.
    InvalidPredicate(String),
    /// Field not found.
    FieldNotFound(String),
    /// Type mismatch.
    TypeMismatch { expected: String, found: String },
    /// Execution error.
    ExecutionError(String),
    /// Plugin load error.
    PluginLoadError(String),
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterError::InvalidPredicate(s) => write!(f, "invalid predicate: {}", s),
            FilterError::FieldNotFound(s) => write!(f, "field not found: {}", s),
            FilterError::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected {}, found {}", expected, found)
            }
            FilterError::ExecutionError(s) => write!(f, "execution error: {}", s),
            FilterError::PluginLoadError(s) => write!(f, "plugin load error: {}", s),
        }
    }
}

impl std::error::Error for FilterError {}

/// Field value types.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
}

impl Value {
    /// Check if value is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::Bytes(b) => !b.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Map(m) => !m.is_empty(),
        }
    }

    /// Get as i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Get as f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Get as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Document/record that can be filtered.
pub trait Filterable {
    /// Get field value by name.
    fn get_field(&self, name: &str) -> Option<Value>;
    
    /// Get document ID.
    fn id(&self) -> u64;
}

/// Simple document implementation.
#[derive(Debug, Clone)]
pub struct Document {
    id: u64,
    fields: HashMap<String, Value>,
}

impl Document {
    /// Create a new document.
    pub fn new(id: u64) -> Self {
        Self {
            id,
            fields: HashMap::new(),
        }
    }

    /// Set a field value.
    pub fn set_field(&mut self, name: impl Into<String>, value: Value) {
        self.fields.insert(name.into(), value);
    }

    /// Create with fields.
    pub fn with_fields(id: u64, fields: HashMap<String, Value>) -> Self {
        Self { id, fields }
    }
}

impl Filterable for Document {
    fn get_field(&self, name: &str) -> Option<Value> {
        self.fields.get(name).cloned()
    }

    fn id(&self) -> u64 {
        self.id
    }
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Contains,
    StartsWith,
    EndsWith,
}

/// Logical operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
    Not,
}

/// Predicate expression tree.
#[derive(Debug, Clone)]
pub enum PredicateExpr {
    /// Literal value.
    Literal(Value),
    /// Field reference.
    Field(String),
    /// Comparison.
    Compare {
        op: CompareOp,
        left: Box<PredicateExpr>,
        right: Box<PredicateExpr>,
    },
    /// Logical operation.
    Logical {
        op: LogicalOp,
        operands: Vec<PredicateExpr>,
    },
    /// In list.
    In {
        value: Box<PredicateExpr>,
        list: Vec<PredicateExpr>,
    },
    /// Between (inclusive).
    Between {
        value: Box<PredicateExpr>,
        low: Box<PredicateExpr>,
        high: Box<PredicateExpr>,
    },
    /// Is null check.
    IsNull(Box<PredicateExpr>),
}

impl PredicateExpr {
    /// Create equality comparison.
    pub fn eq(left: PredicateExpr, right: PredicateExpr) -> Self {
        PredicateExpr::Compare {
            op: CompareOp::Eq,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create field reference.
    pub fn field(name: impl Into<String>) -> Self {
        PredicateExpr::Field(name.into())
    }

    /// Create literal.
    pub fn lit(value: Value) -> Self {
        PredicateExpr::Literal(value)
    }

    /// Create AND expression.
    pub fn and(operands: Vec<PredicateExpr>) -> Self {
        PredicateExpr::Logical {
            op: LogicalOp::And,
            operands,
        }
    }

    /// Create OR expression.
    pub fn or(operands: Vec<PredicateExpr>) -> Self {
        PredicateExpr::Logical {
            op: LogicalOp::Or,
            operands,
        }
    }

    /// Create NOT expression.
    pub fn not(expr: PredicateExpr) -> Self {
        PredicateExpr::Logical {
            op: LogicalOp::Not,
            operands: vec![expr],
        }
    }
}

/// Predicate evaluator.
pub struct PredicateEvaluator {
    /// Root expression.
    expr: PredicateExpr,
    /// Estimated selectivity (0.0 to 1.0).
    selectivity: f64,
}

impl PredicateEvaluator {
    /// Create a new evaluator.
    pub fn new(expr: PredicateExpr) -> Self {
        let selectivity = Self::estimate_selectivity(&expr);
        Self { expr, selectivity }
    }

    /// Evaluate predicate on a document.
    pub fn evaluate<D: Filterable>(&self, doc: &D) -> FilterResult<bool> {
        self.eval_expr(&self.expr, doc).map(|v| v.is_truthy())
    }

    /// Get estimated selectivity.
    pub fn selectivity(&self) -> f64 {
        self.selectivity
    }

    fn eval_expr<D: Filterable>(&self, expr: &PredicateExpr, doc: &D) -> FilterResult<Value> {
        match expr {
            PredicateExpr::Literal(v) => Ok(v.clone()),
            
            PredicateExpr::Field(name) => {
                doc.get_field(name)
                    .ok_or_else(|| FilterError::FieldNotFound(name.clone()))
            }
            
            PredicateExpr::Compare { op, left, right } => {
                let lval = self.eval_expr(left, doc)?;
                let rval = self.eval_expr(right, doc)?;
                Ok(Value::Bool(self.compare(*op, &lval, &rval)))
            }
            
            PredicateExpr::Logical { op, operands } => {
                match op {
                    LogicalOp::And => {
                        for operand in operands {
                            let val = self.eval_expr(operand, doc)?;
                            if !val.is_truthy() {
                                return Ok(Value::Bool(false));
                            }
                        }
                        Ok(Value::Bool(true))
                    }
                    LogicalOp::Or => {
                        for operand in operands {
                            let val = self.eval_expr(operand, doc)?;
                            if val.is_truthy() {
                                return Ok(Value::Bool(true));
                            }
                        }
                        Ok(Value::Bool(false))
                    }
                    LogicalOp::Not => {
                        let val = self.eval_expr(&operands[0], doc)?;
                        Ok(Value::Bool(!val.is_truthy()))
                    }
                }
            }
            
            PredicateExpr::In { value, list } => {
                let val = self.eval_expr(value, doc)?;
                for item in list {
                    let item_val = self.eval_expr(item, doc)?;
                    if self.compare(CompareOp::Eq, &val, &item_val) {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            }
            
            PredicateExpr::Between { value, low, high } => {
                let val = self.eval_expr(value, doc)?;
                let low_val = self.eval_expr(low, doc)?;
                let high_val = self.eval_expr(high, doc)?;
                
                let ge_low = self.compare(CompareOp::Ge, &val, &low_val);
                let le_high = self.compare(CompareOp::Le, &val, &high_val);
                
                Ok(Value::Bool(ge_low && le_high))
            }
            
            PredicateExpr::IsNull(inner) => {
                let val = self.eval_expr(inner, doc).unwrap_or(Value::Null);
                Ok(Value::Bool(matches!(val, Value::Null)))
            }
        }
    }

    fn compare(&self, op: CompareOp, left: &Value, right: &Value) -> bool {
        match op {
            CompareOp::Eq => left == right,
            CompareOp::Ne => left != right,
            CompareOp::Lt => self.ord_compare(left, right) == Some(std::cmp::Ordering::Less),
            CompareOp::Le => {
                matches!(
                    self.ord_compare(left, right),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            }
            CompareOp::Gt => self.ord_compare(left, right) == Some(std::cmp::Ordering::Greater),
            CompareOp::Ge => {
                matches!(
                    self.ord_compare(left, right),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                )
            }
            CompareOp::Contains => self.string_contains(left, right),
            CompareOp::StartsWith => self.string_starts_with(left, right),
            CompareOp::EndsWith => self.string_ends_with(left, right),
        }
    }

    fn ord_compare(&self, left: &Value, right: &Value) -> Option<std::cmp::Ordering> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    fn string_contains(&self, left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::String(a), Value::String(b)) => a.contains(b.as_str()),
            _ => false,
        }
    }

    fn string_starts_with(&self, left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::String(a), Value::String(b)) => a.starts_with(b.as_str()),
            _ => false,
        }
    }

    fn string_ends_with(&self, left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::String(a), Value::String(b)) => a.ends_with(b.as_str()),
            _ => false,
        }
    }

    fn estimate_selectivity(expr: &PredicateExpr) -> f64 {
        match expr {
            PredicateExpr::Literal(Value::Bool(true)) => 1.0,
            PredicateExpr::Literal(Value::Bool(false)) => 0.0,
            PredicateExpr::Literal(_) => 0.5,
            PredicateExpr::Field(_) => 0.5,
            PredicateExpr::Compare { op, .. } => {
                match op {
                    CompareOp::Eq => 0.1,      // Equality is selective
                    CompareOp::Ne => 0.9,      // Not-equal is not selective
                    CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => 0.3,
                    CompareOp::Contains | CompareOp::StartsWith | CompareOp::EndsWith => 0.2,
                }
            }
            PredicateExpr::Logical { op, operands } => {
                match op {
                    LogicalOp::And => {
                        operands.iter().map(Self::estimate_selectivity).product()
                    }
                    LogicalOp::Or => {
                        let sels: Vec<f64> = operands.iter()
                            .map(Self::estimate_selectivity)
                            .collect();
                        // P(A or B) = P(A) + P(B) - P(A)*P(B)
                        sels.iter().fold(0.0, |acc, &s| acc + s - acc * s)
                    }
                    LogicalOp::Not => {
                        1.0 - Self::estimate_selectivity(&operands[0])
                    }
                }
            }
            PredicateExpr::In { list, .. } => {
                (list.len() as f64 * 0.1).min(0.9)
            }
            PredicateExpr::Between { .. } => 0.2,
            PredicateExpr::IsNull(_) => 0.05,
        }
    }
}

/// Trait for filter plugins.
pub trait FilterPlugin: Send + Sync {
    /// Plugin name.
    fn name(&self) -> &str;
    
    /// Evaluate filter on document.
    fn evaluate(&self, doc: &dyn Filterable) -> FilterResult<bool>;
    
    /// Estimated selectivity.
    fn selectivity(&self) -> f64 {
        0.5
    }
    
    /// Estimated cost per evaluation (relative units).
    fn cost(&self) -> f64 {
        1.0
    }
}

/// Native Rust filter plugin.
pub struct NativeFilter<F>
where
    F: Fn(&dyn Filterable) -> bool + Send + Sync,
{
    name: String,
    filter_fn: F,
    selectivity: f64,
}

impl<F> NativeFilter<F>
where
    F: Fn(&dyn Filterable) -> bool + Send + Sync,
{
    /// Create a new native filter.
    pub fn new(name: impl Into<String>, filter_fn: F) -> Self {
        Self {
            name: name.into(),
            filter_fn,
            selectivity: 0.5,
        }
    }

    /// Set estimated selectivity.
    pub fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.selectivity = selectivity;
        self
    }
}

impl<F> FilterPlugin for NativeFilter<F>
where
    F: Fn(&dyn Filterable) -> bool + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, doc: &dyn Filterable) -> FilterResult<bool> {
        Ok((self.filter_fn)(doc))
    }

    fn selectivity(&self) -> f64 {
        self.selectivity
    }

    fn cost(&self) -> f64 {
        0.5 // Native is cheap
    }
}

/// Filter pipeline that orders filters by cost-effectiveness.
pub struct FilterPipeline {
    /// Filters in execution order.
    filters: Vec<Arc<dyn FilterPlugin>>,
}

impl FilterPipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a filter.
    pub fn add(&mut self, filter: Arc<dyn FilterPlugin>) {
        self.filters.push(filter);
        self.reorder();
    }

    /// Reorder filters by cost-effectiveness.
    fn reorder(&mut self) {
        // Order by (1 - selectivity) / cost (most selective and cheapest first)
        self.filters.sort_by(|a, b| {
            let score_a = (1.0 - a.selectivity()) / a.cost();
            let score_b = (1.0 - b.selectivity()) / b.cost();
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Evaluate all filters on a document.
    pub fn evaluate(&self, doc: &dyn Filterable) -> FilterResult<bool> {
        for filter in &self.filters {
            if !filter.evaluate(doc)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Filter a batch of documents.
    pub fn filter_batch<'a, D: Filterable>(&self, docs: &'a [D]) -> Vec<&'a D> {
        docs.iter()
            .filter(|doc| self.evaluate(*doc).unwrap_or(false))
            .collect()
    }

    /// Get number of filters.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Estimated combined selectivity.
    pub fn combined_selectivity(&self) -> f64 {
        self.filters.iter().map(|f| f.selectivity()).product()
    }
}

impl Default for FilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Projection specification.
#[derive(Debug, Clone)]
pub struct Projection {
    /// Fields to include.
    pub include: Vec<String>,
    /// Fields to exclude.
    pub exclude: Vec<String>,
}

impl Projection {
    /// Create include-only projection.
    pub fn include(fields: Vec<String>) -> Self {
        Self {
            include: fields,
            exclude: Vec::new(),
        }
    }

    /// Create exclude projection.
    pub fn exclude(fields: Vec<String>) -> Self {
        Self {
            include: Vec::new(),
            exclude: fields,
        }
    }

    /// Check if field is included.
    pub fn includes(&self, field: &str) -> bool {
        if !self.include.is_empty() {
            self.include.iter().any(|f| f == field)
        } else if !self.exclude.is_empty() {
            !self.exclude.iter().any(|f| f == field)
        } else {
            true // Include all by default
        }
    }

    /// Apply projection to document.
    pub fn apply(&self, doc: &Document) -> Document {
        let mut result = Document::new(doc.id);
        for (name, value) in &doc.fields {
            if self.includes(name) {
                result.set_field(name.clone(), value.clone());
            }
        }
        result
    }
}

// ============================================================================
// WASM Plugin Support
// ============================================================================

/// WASM plugin configuration.
#[derive(Debug, Clone)]
pub struct WasmPluginConfig {
    /// Maximum memory in bytes (default 16MB).
    pub max_memory: usize,
    /// Maximum fuel/instructions per evaluation (default 100k).
    pub max_fuel: u64,
    /// Timeout in milliseconds (default 10ms).
    pub timeout_ms: u64,
    /// Allow network access.
    pub allow_network: bool,
    /// Allow file system access.
    pub allow_filesystem: bool,
}

impl Default for WasmPluginConfig {
    fn default() -> Self {
        Self {
            max_memory: 16 * 1024 * 1024, // 16MB
            max_fuel: 100_000,
            timeout_ms: 10,
            allow_network: false,
            allow_filesystem: false,
        }
    }
}

/// WASM module bytecode.
#[derive(Debug, Clone)]
pub struct WasmModule {
    /// Module name.
    pub name: String,
    /// WASM bytecode.
    pub bytecode: Vec<u8>,
    /// Configuration.
    pub config: WasmPluginConfig,
}

impl WasmModule {
    /// Create a new WASM module.
    pub fn new(name: impl Into<String>, bytecode: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            bytecode,
            config: WasmPluginConfig::default(),
        }
    }

    /// Set configuration.
    pub fn with_config(mut self, config: WasmPluginConfig) -> Self {
        self.config = config;
        self
    }

    /// Validate WASM bytecode (basic check).
    pub fn validate(&self) -> FilterResult<()> {
        // Check WASM magic number: \0asm
        if self.bytecode.len() < 8 {
            return Err(FilterError::PluginLoadError(
                "WASM bytecode too short".to_string()
            ));
        }
        if &self.bytecode[0..4] != b"\0asm" {
            return Err(FilterError::PluginLoadError(
                "Invalid WASM magic number".to_string()
            ));
        }
        Ok(())
    }

    /// Get bytecode size.
    pub fn size(&self) -> usize {
        self.bytecode.len()
    }
}

/// WASM filter plugin (sandboxed execution).
///
/// This provides a safe, sandboxed environment for user-defined filters.
/// The actual WASM runtime (wasmtime, wasmer, etc.) would be integrated here.
pub struct WasmFilter {
    /// Module definition.
    module: WasmModule,
    /// Execution statistics.
    stats: WasmFilterStats,
}

/// WASM filter execution statistics.
#[derive(Debug, Clone, Default)]
pub struct WasmFilterStats {
    /// Total evaluations.
    pub evaluations: u64,
    /// Total fuel consumed.
    pub fuel_consumed: u64,
    /// Total time spent (microseconds).
    pub time_us: u64,
    /// Evaluation errors.
    pub errors: u64,
}

impl WasmFilter {
    /// Create a new WASM filter.
    pub fn new(module: WasmModule) -> FilterResult<Self> {
        module.validate()?;
        Ok(Self {
            module,
            stats: WasmFilterStats::default(),
        })
    }

    /// Get module name.
    pub fn name(&self) -> &str {
        &self.module.name
    }

    /// Get configuration.
    pub fn config(&self) -> &WasmPluginConfig {
        &self.module.config
    }

    /// Get execution statistics.
    pub fn stats(&self) -> &WasmFilterStats {
        &self.stats
    }

    /// Evaluate filter on a document.
    ///
    /// In a full implementation, this would:
    /// 1. Serialize document to WASM-compatible format
    /// 2. Call WASM function with fuel metering
    /// 3. Deserialize result
    pub fn evaluate(&mut self, doc: &dyn Filterable) -> FilterResult<bool> {
        self.stats.evaluations += 1;
        
        // Placeholder: actual WASM runtime integration would go here
        // For now, return a default based on document existence
        let _ = doc.id();
        
        // Simulate some fuel consumption
        self.stats.fuel_consumed += 100;
        
        // In production, this would execute the WASM module
        // with proper sandboxing via wasmtime/wasmer
        Ok(true)
    }
}

impl FilterPlugin for WasmFilter {
    fn name(&self) -> &str {
        &self.module.name
    }

    fn evaluate(&self, doc: &dyn Filterable) -> FilterResult<bool> {
        // Note: stats tracking would require interior mutability in production
        let _ = doc.id();
        Ok(true)
    }

    fn selectivity(&self) -> f64 {
        0.5 // Conservative estimate
    }

    fn cost(&self) -> f64 {
        5.0 // WASM has higher overhead than native
    }
}

/// WASM plugin registry for managing loaded modules.
pub struct WasmPluginRegistry {
    /// Loaded plugins by name.
    plugins: HashMap<String, Arc<WasmFilter>>,
    /// Maximum plugins allowed.
    max_plugins: usize,
}

impl WasmPluginRegistry {
    /// Create a new registry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            max_plugins: 100,
        }
    }

    /// Create with capacity limit.
    pub fn with_max_plugins(max_plugins: usize) -> Self {
        Self {
            plugins: HashMap::new(),
            max_plugins,
        }
    }

    /// Register a WASM module.
    pub fn register(&mut self, module: WasmModule) -> FilterResult<()> {
        if self.plugins.len() >= self.max_plugins {
            return Err(FilterError::PluginLoadError(
                "plugin registry full".to_string()
            ));
        }

        let name = module.name.clone();
        let filter = WasmFilter::new(module)?;
        self.plugins.insert(name, Arc::new(filter));
        Ok(())
    }

    /// Get a plugin by name.
    pub fn get(&self, name: &str) -> Option<Arc<WasmFilter>> {
        self.plugins.get(name).cloned()
    }

    /// Remove a plugin.
    pub fn remove(&mut self, name: &str) -> bool {
        self.plugins.remove(name).is_some()
    }

    /// List all plugin names.
    pub fn list(&self) -> Vec<&str> {
        self.plugins.keys().map(|s| s.as_str()).collect()
    }

    /// Get plugin count.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

impl Default for WasmPluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> Document {
        let mut doc = Document::new(1);
        doc.set_field("name", Value::String("Alice".to_string()));
        doc.set_field("age", Value::Int(30));
        doc.set_field("score", Value::Float(95.5));
        doc.set_field("active", Value::Bool(true));
        doc
    }

    #[test]
    fn test_value_truthy() {
        assert!(!Value::Null.is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!(Value::Bool(true).is_truthy());
        assert!(!Value::Int(0).is_truthy());
        assert!(Value::Int(1).is_truthy());
        assert!(!Value::String("".to_string()).is_truthy());
        assert!(Value::String("hello".to_string()).is_truthy());
    }

    #[test]
    fn test_simple_equality() {
        let expr = PredicateExpr::eq(
            PredicateExpr::field("age"),
            PredicateExpr::lit(Value::Int(30)),
        );
        let eval = PredicateEvaluator::new(expr);
        
        let doc = sample_doc();
        assert!(eval.evaluate(&doc).unwrap());
    }

    #[test]
    fn test_comparison() {
        let expr = PredicateExpr::Compare {
            op: CompareOp::Gt,
            left: Box::new(PredicateExpr::field("score")),
            right: Box::new(PredicateExpr::lit(Value::Float(90.0))),
        };
        let eval = PredicateEvaluator::new(expr);
        
        let doc = sample_doc();
        assert!(eval.evaluate(&doc).unwrap());
    }

    #[test]
    fn test_logical_and() {
        let expr = PredicateExpr::and(vec![
            PredicateExpr::eq(
                PredicateExpr::field("active"),
                PredicateExpr::lit(Value::Bool(true)),
            ),
            PredicateExpr::Compare {
                op: CompareOp::Ge,
                left: Box::new(PredicateExpr::field("age")),
                right: Box::new(PredicateExpr::lit(Value::Int(18))),
            },
        ]);
        let eval = PredicateEvaluator::new(expr);
        
        let doc = sample_doc();
        assert!(eval.evaluate(&doc).unwrap());
    }

    #[test]
    fn test_logical_or() {
        let expr = PredicateExpr::or(vec![
            PredicateExpr::eq(
                PredicateExpr::field("name"),
                PredicateExpr::lit(Value::String("Bob".to_string())),
            ),
            PredicateExpr::eq(
                PredicateExpr::field("name"),
                PredicateExpr::lit(Value::String("Alice".to_string())),
            ),
        ]);
        let eval = PredicateEvaluator::new(expr);
        
        let doc = sample_doc();
        assert!(eval.evaluate(&doc).unwrap());
    }

    #[test]
    fn test_string_contains() {
        let expr = PredicateExpr::Compare {
            op: CompareOp::Contains,
            left: Box::new(PredicateExpr::field("name")),
            right: Box::new(PredicateExpr::lit(Value::String("lic".to_string()))),
        };
        let eval = PredicateEvaluator::new(expr);
        
        let doc = sample_doc();
        assert!(eval.evaluate(&doc).unwrap());
    }

    #[test]
    fn test_between() {
        let expr = PredicateExpr::Between {
            value: Box::new(PredicateExpr::field("age")),
            low: Box::new(PredicateExpr::lit(Value::Int(25))),
            high: Box::new(PredicateExpr::lit(Value::Int(35))),
        };
        let eval = PredicateEvaluator::new(expr);
        
        let doc = sample_doc();
        assert!(eval.evaluate(&doc).unwrap());
    }

    #[test]
    fn test_native_filter() {
        let filter = NativeFilter::new("age_filter", |doc| {
            doc.get_field("age")
                .and_then(|v| v.as_int())
                .map(|age| age >= 18)
                .unwrap_or(false)
        });

        let doc = sample_doc();
        assert!(filter.evaluate(&doc).unwrap());
    }

    #[test]
    fn test_filter_pipeline() {
        let mut pipeline = FilterPipeline::new();
        
        let filter1 = Arc::new(NativeFilter::new("active_check", |doc| {
            doc.get_field("active")
                .map(|v| v.is_truthy())
                .unwrap_or(false)
        }).with_selectivity(0.8));
        
        let filter2 = Arc::new(NativeFilter::new("score_check", |doc| {
            doc.get_field("score")
                .and_then(|v| v.as_float())
                .map(|s| s > 50.0)
                .unwrap_or(false)
        }).with_selectivity(0.3));

        pipeline.add(filter1);
        pipeline.add(filter2);

        let doc = sample_doc();
        assert!(pipeline.evaluate(&doc).unwrap());
    }

    #[test]
    fn test_selectivity_estimation() {
        let expr = PredicateExpr::and(vec![
            PredicateExpr::eq(
                PredicateExpr::field("x"),
                PredicateExpr::lit(Value::Int(1)),
            ),
            PredicateExpr::eq(
                PredicateExpr::field("y"),
                PredicateExpr::lit(Value::Int(2)),
            ),
        ]);
        let eval = PredicateEvaluator::new(expr);
        
        // AND of two equality checks: 0.1 * 0.1 = 0.01
        assert!((eval.selectivity() - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_projection_include() {
        let proj = Projection::include(vec!["name".to_string(), "age".to_string()]);
        
        assert!(proj.includes("name"));
        assert!(proj.includes("age"));
        assert!(!proj.includes("score"));
    }

    #[test]
    fn test_projection_apply() {
        let proj = Projection::include(vec!["name".to_string()]);
        let doc = sample_doc();
        
        let result = proj.apply(&doc);
        assert!(result.get_field("name").is_some());
        assert!(result.get_field("age").is_none());
    }

    #[test]
    fn test_filter_batch() {
        let mut pipeline = FilterPipeline::new();
        pipeline.add(Arc::new(NativeFilter::new("age_filter", |doc| {
            doc.get_field("age")
                .and_then(|v| v.as_int())
                .map(|age| age >= 25)
                .unwrap_or(false)
        })));

        let mut docs = Vec::new();
        for i in 0..10 {
            let mut doc = Document::new(i);
            doc.set_field("age", Value::Int((i * 5) as i64));
            docs.push(doc);
        }

        let filtered = pipeline.filter_batch(&docs);
        // Ages 0,5,10,15,20,25,30,35,40,45 -> only 25,30,35,40,45 pass
        assert_eq!(filtered.len(), 5);
    }

    // ========================================================================
    // WASM Plugin Tests
    // ========================================================================

    #[test]
    fn test_wasm_plugin_config_default() {
        let config = WasmPluginConfig::default();
        assert_eq!(config.max_memory, 16 * 1024 * 1024);
        assert_eq!(config.max_fuel, 100_000);
        assert_eq!(config.timeout_ms, 10);
        assert!(!config.allow_network);
        assert!(!config.allow_filesystem);
    }

    #[test]
    fn test_wasm_module_validation() {
        // Valid WASM magic number
        let valid_wasm = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        let module = WasmModule::new("test", valid_wasm);
        assert!(module.validate().is_ok());

        // Invalid - too short
        let short = vec![0x00, 0x61];
        let module = WasmModule::new("short", short);
        assert!(module.validate().is_err());

        // Invalid magic number
        let invalid = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let module = WasmModule::new("invalid", invalid);
        assert!(module.validate().is_err());
    }

    #[test]
    fn test_wasm_filter_creation() {
        let wasm = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        let module = WasmModule::new("test_filter", wasm);
        let filter = WasmFilter::new(module).unwrap();

        assert_eq!(filter.name(), "test_filter");
        assert_eq!(filter.stats().evaluations, 0);
    }

    #[test]
    fn test_wasm_registry() {
        let mut registry = WasmPluginRegistry::new();
        assert!(registry.is_empty());

        let wasm = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        registry.register(WasmModule::new("filter1", wasm.clone())).unwrap();
        registry.register(WasmModule::new("filter2", wasm.clone())).unwrap();

        assert_eq!(registry.len(), 2);
        assert!(registry.get("filter1").is_some());
        assert!(registry.get("nonexistent").is_none());

        let names = registry.list();
        assert!(names.contains(&"filter1"));
        assert!(names.contains(&"filter2"));

        assert!(registry.remove("filter1"));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_wasm_registry_limit() {
        let mut registry = WasmPluginRegistry::with_max_plugins(2);
        let wasm = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

        registry.register(WasmModule::new("p1", wasm.clone())).unwrap();
        registry.register(WasmModule::new("p2", wasm.clone())).unwrap();
        
        // Third should fail
        let result = registry.register(WasmModule::new("p3", wasm));
        assert!(result.is_err());
    }

    #[test]
    fn test_wasm_filter_as_plugin() {
        let wasm = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        let module = WasmModule::new("plugin", wasm);
        let filter = WasmFilter::new(module).unwrap();

        // Test FilterPlugin trait
        assert_eq!(filter.name(), "plugin");
        assert!(filter.cost() > 1.0); // WASM is more expensive

        let doc = sample_doc();
        assert!(filter.evaluate(&doc).is_ok());
    }
}

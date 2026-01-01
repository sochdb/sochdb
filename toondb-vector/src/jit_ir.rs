// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! JIT Compilation for Small IR
//!
//! This module provides a minimal SSA-like IR for filter expressions
//! that can be JIT-compiled for hot paths.
//!
//! # Problem
//!
//! Interpreted filter evaluation has overhead:
//! - Function call per operation
//! - Branch misprediction from expression dispatch
//! - No optimization across operations
//!
//! # Solution
//!
//! JIT compilation for hot filters:
//! 1. Define simple SSA-like IR
//! 2. Compile to native code (via Cranelift when available)
//! 3. Cache compiled functions
//! 4. Fall back to interpreter for cold paths
//!
//! # IR Design
//!
//! - Three-address code style
//! - Explicit types (i64, f64, bool, ptr)
//! - No control flow (single basic block)
//! - Used for simple filter expressions
//!
//! # Example
//!
//! Filter: age >= 18 AND score > 50.0
//!
//! ```text
//! v0 = load_i64 field:0     ; load age
//! v1 = const_i64 18
//! v2 = gte_i64 v0, v1       ; age >= 18
//! v3 = load_f64 field:1     ; load score
//! v4 = const_f64 50.0
//! v5 = gt_f64 v3, v4        ; score > 50.0
//! v6 = and v2, v5           ; AND
//! ret v6
//! ```

use std::collections::HashMap;
use std::sync::Arc;

/// Value type in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrType {
    Bool,
    I64,
    F64,
    Ptr,
}

impl IrType {
    /// Size in bytes.
    pub fn size(&self) -> usize {
        match self {
            IrType::Bool => 1,
            IrType::I64 | IrType::F64 | IrType::Ptr => 8,
        }
    }
}

/// Virtual register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

impl Reg {
    pub fn new(id: u32) -> Self {
        Reg(id)
    }
}

/// IR instruction.
#[derive(Debug, Clone)]
pub enum IrInst {
    // Constants
    ConstBool(Reg, bool),
    ConstI64(Reg, i64),
    ConstF64(Reg, f64),
    
    // Load from field (field index)
    LoadI64(Reg, u32),
    LoadF64(Reg, u32),
    LoadBool(Reg, u32),
    
    // Integer comparisons
    EqI64(Reg, Reg, Reg),
    NeI64(Reg, Reg, Reg),
    LtI64(Reg, Reg, Reg),
    LeI64(Reg, Reg, Reg),
    GtI64(Reg, Reg, Reg),
    GeI64(Reg, Reg, Reg),
    
    // Float comparisons
    EqF64(Reg, Reg, Reg),
    NeF64(Reg, Reg, Reg),
    LtF64(Reg, Reg, Reg),
    LeF64(Reg, Reg, Reg),
    GtF64(Reg, Reg, Reg),
    GeF64(Reg, Reg, Reg),
    
    // Logical
    And(Reg, Reg, Reg),
    Or(Reg, Reg, Reg),
    Not(Reg, Reg),
    
    // Arithmetic (for expression evaluation)
    AddI64(Reg, Reg, Reg),
    SubI64(Reg, Reg, Reg),
    MulI64(Reg, Reg, Reg),
    AddF64(Reg, Reg, Reg),
    SubF64(Reg, Reg, Reg),
    MulF64(Reg, Reg, Reg),
    DivF64(Reg, Reg, Reg),
    
    // Type conversions
    I64ToF64(Reg, Reg),
    F64ToI64(Reg, Reg),
    
    // Return result
    Ret(Reg),
}

impl IrInst {
    /// Get destination register if any.
    pub fn dest(&self) -> Option<Reg> {
        match self {
            IrInst::ConstBool(r, _) |
            IrInst::ConstI64(r, _) |
            IrInst::ConstF64(r, _) |
            IrInst::LoadI64(r, _) |
            IrInst::LoadF64(r, _) |
            IrInst::LoadBool(r, _) |
            IrInst::EqI64(r, _, _) |
            IrInst::NeI64(r, _, _) |
            IrInst::LtI64(r, _, _) |
            IrInst::LeI64(r, _, _) |
            IrInst::GtI64(r, _, _) |
            IrInst::GeI64(r, _, _) |
            IrInst::EqF64(r, _, _) |
            IrInst::NeF64(r, _, _) |
            IrInst::LtF64(r, _, _) |
            IrInst::LeF64(r, _, _) |
            IrInst::GtF64(r, _, _) |
            IrInst::GeF64(r, _, _) |
            IrInst::And(r, _, _) |
            IrInst::Or(r, _, _) |
            IrInst::Not(r, _) |
            IrInst::AddI64(r, _, _) |
            IrInst::SubI64(r, _, _) |
            IrInst::MulI64(r, _, _) |
            IrInst::AddF64(r, _, _) |
            IrInst::SubF64(r, _, _) |
            IrInst::MulF64(r, _, _) |
            IrInst::DivF64(r, _, _) |
            IrInst::I64ToF64(r, _) |
            IrInst::F64ToI64(r, _) => Some(*r),
            IrInst::Ret(_) => None,
        }
    }

    /// Get source registers.
    pub fn sources(&self) -> Vec<Reg> {
        match self {
            IrInst::ConstBool(_, _) |
            IrInst::ConstI64(_, _) |
            IrInst::ConstF64(_, _) |
            IrInst::LoadI64(_, _) |
            IrInst::LoadF64(_, _) |
            IrInst::LoadBool(_, _) => vec![],
            
            IrInst::Not(_, src) |
            IrInst::I64ToF64(_, src) |
            IrInst::F64ToI64(_, src) |
            IrInst::Ret(src) => vec![*src],
            
            IrInst::EqI64(_, a, b) |
            IrInst::NeI64(_, a, b) |
            IrInst::LtI64(_, a, b) |
            IrInst::LeI64(_, a, b) |
            IrInst::GtI64(_, a, b) |
            IrInst::GeI64(_, a, b) |
            IrInst::EqF64(_, a, b) |
            IrInst::NeF64(_, a, b) |
            IrInst::LtF64(_, a, b) |
            IrInst::LeF64(_, a, b) |
            IrInst::GtF64(_, a, b) |
            IrInst::GeF64(_, a, b) |
            IrInst::And(_, a, b) |
            IrInst::Or(_, a, b) |
            IrInst::AddI64(_, a, b) |
            IrInst::SubI64(_, a, b) |
            IrInst::MulI64(_, a, b) |
            IrInst::AddF64(_, a, b) |
            IrInst::SubF64(_, a, b) |
            IrInst::MulF64(_, a, b) |
            IrInst::DivF64(_, a, b) => vec![*a, *b],
        }
    }
}

/// IR function (single basic block).
#[derive(Debug, Clone)]
pub struct IrFunction {
    /// Instructions.
    pub instructions: Vec<IrInst>,
    /// Register types.
    pub reg_types: HashMap<Reg, IrType>,
    /// Field types (for load instructions).
    pub field_types: Vec<IrType>,
    /// Number of registers used.
    pub num_regs: u32,
}

impl IrFunction {
    /// Create a new empty function.
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            reg_types: HashMap::new(),
            field_types: Vec::new(),
            num_regs: 0,
        }
    }

    /// Allocate a new register.
    pub fn alloc_reg(&mut self, ty: IrType) -> Reg {
        let reg = Reg(self.num_regs);
        self.num_regs += 1;
        self.reg_types.insert(reg, ty);
        reg
    }

    /// Add a field type.
    pub fn add_field(&mut self, ty: IrType) -> u32 {
        let idx = self.field_types.len() as u32;
        self.field_types.push(ty);
        idx
    }

    /// Add an instruction.
    pub fn push(&mut self, inst: IrInst) {
        self.instructions.push(inst);
    }

    /// Get instruction count.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Validate the function.
    pub fn validate(&self) -> Result<(), String> {
        // Check that all used registers are defined
        let mut defined: std::collections::HashSet<Reg> = std::collections::HashSet::new();
        
        for inst in &self.instructions {
            // Check sources are defined
            for src in inst.sources() {
                if !defined.contains(&src) {
                    return Err(format!("register {:?} used before definition", src));
                }
            }
            
            // Mark destination as defined
            if let Some(dest) = inst.dest() {
                defined.insert(dest);
            }
        }

        // Check that function ends with Ret
        if let Some(last) = self.instructions.last() {
            if !matches!(last, IrInst::Ret(_)) {
                return Err("function must end with Ret".to_string());
            }
        } else {
            return Err("function is empty".to_string());
        }

        Ok(())
    }
}

impl Default for IrFunction {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime value for interpreter.
#[derive(Debug, Clone, Copy)]
pub enum RtValue {
    Bool(bool),
    I64(i64),
    F64(f64),
}

impl RtValue {
    pub fn as_bool(&self) -> bool {
        match self {
            RtValue::Bool(b) => *b,
            RtValue::I64(i) => *i != 0,
            RtValue::F64(f) => *f != 0.0,
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            RtValue::Bool(b) => if *b { 1 } else { 0 },
            RtValue::I64(i) => *i,
            RtValue::F64(f) => *f as i64,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            RtValue::Bool(b) => if *b { 1.0 } else { 0.0 },
            RtValue::I64(i) => *i as f64,
            RtValue::F64(f) => *f,
        }
    }
}

/// Field accessor for runtime.
pub trait FieldAccess {
    fn get_i64(&self, field: u32) -> i64;
    fn get_f64(&self, field: u32) -> f64;
    fn get_bool(&self, field: u32) -> bool;
}

/// Simple array-based field storage.
pub struct ArrayFields {
    i64_fields: Vec<i64>,
    f64_fields: Vec<f64>,
    bool_fields: Vec<bool>,
}

impl ArrayFields {
    pub fn new() -> Self {
        Self {
            i64_fields: Vec::new(),
            f64_fields: Vec::new(),
            bool_fields: Vec::new(),
        }
    }

    pub fn set_i64(&mut self, idx: usize, value: i64) {
        if idx >= self.i64_fields.len() {
            self.i64_fields.resize(idx + 1, 0);
        }
        self.i64_fields[idx] = value;
    }

    pub fn set_f64(&mut self, idx: usize, value: f64) {
        if idx >= self.f64_fields.len() {
            self.f64_fields.resize(idx + 1, 0.0);
        }
        self.f64_fields[idx] = value;
    }

    pub fn set_bool(&mut self, idx: usize, value: bool) {
        if idx >= self.bool_fields.len() {
            self.bool_fields.resize(idx + 1, false);
        }
        self.bool_fields[idx] = value;
    }
}

impl Default for ArrayFields {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldAccess for ArrayFields {
    fn get_i64(&self, field: u32) -> i64 {
        self.i64_fields.get(field as usize).copied().unwrap_or(0)
    }

    fn get_f64(&self, field: u32) -> f64 {
        self.f64_fields.get(field as usize).copied().unwrap_or(0.0)
    }

    fn get_bool(&self, field: u32) -> bool {
        self.bool_fields.get(field as usize).copied().unwrap_or(false)
    }
}

/// IR interpreter (fallback when JIT not available).
pub struct IrInterpreter {
    /// The function to interpret.
    function: IrFunction,
}

impl IrInterpreter {
    /// Create a new interpreter.
    pub fn new(function: IrFunction) -> Result<Self, String> {
        function.validate()?;
        Ok(Self { function })
    }

    /// Execute the function.
    pub fn execute<F: FieldAccess>(&self, fields: &F) -> bool {
        let mut regs: Vec<RtValue> = vec![RtValue::Bool(false); self.function.num_regs as usize];

        for inst in &self.function.instructions {
            match inst {
                IrInst::ConstBool(dest, val) => {
                    regs[dest.0 as usize] = RtValue::Bool(*val);
                }
                IrInst::ConstI64(dest, val) => {
                    regs[dest.0 as usize] = RtValue::I64(*val);
                }
                IrInst::ConstF64(dest, val) => {
                    regs[dest.0 as usize] = RtValue::F64(*val);
                }
                IrInst::LoadI64(dest, field) => {
                    regs[dest.0 as usize] = RtValue::I64(fields.get_i64(*field));
                }
                IrInst::LoadF64(dest, field) => {
                    regs[dest.0 as usize] = RtValue::F64(fields.get_f64(*field));
                }
                IrInst::LoadBool(dest, field) => {
                    regs[dest.0 as usize] = RtValue::Bool(fields.get_bool(*field));
                }
                IrInst::EqI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64() == regs[b.0 as usize].as_i64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::NeI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64() != regs[b.0 as usize].as_i64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::LtI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64() < regs[b.0 as usize].as_i64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::LeI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64() <= regs[b.0 as usize].as_i64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::GtI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64() > regs[b.0 as usize].as_i64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::GeI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64() >= regs[b.0 as usize].as_i64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::EqF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() == regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::NeF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() != regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::LtF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() < regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::LeF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() <= regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::GtF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() > regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::GeF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() >= regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::And(dest, a, b) => {
                    let result = regs[a.0 as usize].as_bool() && regs[b.0 as usize].as_bool();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::Or(dest, a, b) => {
                    let result = regs[a.0 as usize].as_bool() || regs[b.0 as usize].as_bool();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::Not(dest, src) => {
                    let result = !regs[src.0 as usize].as_bool();
                    regs[dest.0 as usize] = RtValue::Bool(result);
                }
                IrInst::AddI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64().wrapping_add(regs[b.0 as usize].as_i64());
                    regs[dest.0 as usize] = RtValue::I64(result);
                }
                IrInst::SubI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64().wrapping_sub(regs[b.0 as usize].as_i64());
                    regs[dest.0 as usize] = RtValue::I64(result);
                }
                IrInst::MulI64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_i64().wrapping_mul(regs[b.0 as usize].as_i64());
                    regs[dest.0 as usize] = RtValue::I64(result);
                }
                IrInst::AddF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() + regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::F64(result);
                }
                IrInst::SubF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() - regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::F64(result);
                }
                IrInst::MulF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() * regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::F64(result);
                }
                IrInst::DivF64(dest, a, b) => {
                    let result = regs[a.0 as usize].as_f64() / regs[b.0 as usize].as_f64();
                    regs[dest.0 as usize] = RtValue::F64(result);
                }
                IrInst::I64ToF64(dest, src) => {
                    regs[dest.0 as usize] = RtValue::F64(regs[src.0 as usize].as_i64() as f64);
                }
                IrInst::F64ToI64(dest, src) => {
                    regs[dest.0 as usize] = RtValue::I64(regs[src.0 as usize].as_f64() as i64);
                }
                IrInst::Ret(src) => {
                    return regs[src.0 as usize].as_bool();
                }
            }
        }

        false
    }
}

/// IR builder helper.
pub struct IrBuilder {
    function: IrFunction,
}

impl IrBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            function: IrFunction::new(),
        }
    }

    /// Define a field.
    pub fn define_field(&mut self, ty: IrType) -> u32 {
        self.function.add_field(ty)
    }

    /// Load an i64 field.
    pub fn load_i64(&mut self, field: u32) -> Reg {
        let reg = self.function.alloc_reg(IrType::I64);
        self.function.push(IrInst::LoadI64(reg, field));
        reg
    }

    /// Load an f64 field.
    pub fn load_f64(&mut self, field: u32) -> Reg {
        let reg = self.function.alloc_reg(IrType::F64);
        self.function.push(IrInst::LoadF64(reg, field));
        reg
    }

    /// Create i64 constant.
    pub fn const_i64(&mut self, val: i64) -> Reg {
        let reg = self.function.alloc_reg(IrType::I64);
        self.function.push(IrInst::ConstI64(reg, val));
        reg
    }

    /// Create f64 constant.
    pub fn const_f64(&mut self, val: f64) -> Reg {
        let reg = self.function.alloc_reg(IrType::F64);
        self.function.push(IrInst::ConstF64(reg, val));
        reg
    }

    /// Create bool constant.
    pub fn const_bool(&mut self, val: bool) -> Reg {
        let reg = self.function.alloc_reg(IrType::Bool);
        self.function.push(IrInst::ConstBool(reg, val));
        reg
    }

    /// Greater-than-or-equal for i64.
    pub fn ge_i64(&mut self, a: Reg, b: Reg) -> Reg {
        let reg = self.function.alloc_reg(IrType::Bool);
        self.function.push(IrInst::GeI64(reg, a, b));
        reg
    }

    /// Greater-than for i64.
    pub fn gt_i64(&mut self, a: Reg, b: Reg) -> Reg {
        let reg = self.function.alloc_reg(IrType::Bool);
        self.function.push(IrInst::GtI64(reg, a, b));
        reg
    }

    /// Less-than for i64.
    pub fn lt_i64(&mut self, a: Reg, b: Reg) -> Reg {
        let reg = self.function.alloc_reg(IrType::Bool);
        self.function.push(IrInst::LtI64(reg, a, b));
        reg
    }

    /// Greater-than for f64.
    pub fn gt_f64(&mut self, a: Reg, b: Reg) -> Reg {
        let reg = self.function.alloc_reg(IrType::Bool);
        self.function.push(IrInst::GtF64(reg, a, b));
        reg
    }

    /// Logical AND.
    pub fn and(&mut self, a: Reg, b: Reg) -> Reg {
        let reg = self.function.alloc_reg(IrType::Bool);
        self.function.push(IrInst::And(reg, a, b));
        reg
    }

    /// Logical OR.
    pub fn or(&mut self, a: Reg, b: Reg) -> Reg {
        let reg = self.function.alloc_reg(IrType::Bool);
        self.function.push(IrInst::Or(reg, a, b));
        reg
    }

    /// Logical NOT.
    pub fn not(&mut self, a: Reg) -> Reg {
        let reg = self.function.alloc_reg(IrType::Bool);
        self.function.push(IrInst::Not(reg, a));
        reg
    }

    /// Return result.
    pub fn ret(&mut self, reg: Reg) {
        self.function.push(IrInst::Ret(reg));
    }

    /// Build the function.
    pub fn build(self) -> IrFunction {
        self.function
    }
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled filter that can be executed.
pub struct CompiledFilter {
    /// The underlying interpreter (or JIT code).
    interpreter: IrInterpreter,
    /// Execution count for hot-path detection.
    exec_count: std::sync::atomic::AtomicU64,
}

impl CompiledFilter {
    /// Compile a filter function.
    pub fn compile(function: IrFunction) -> Result<Self, String> {
        let interpreter = IrInterpreter::new(function)?;
        Ok(Self {
            interpreter,
            exec_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Execute the filter.
    pub fn execute<F: FieldAccess>(&self, fields: &F) -> bool {
        self.exec_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.interpreter.execute(fields)
    }

    /// Get execution count.
    pub fn exec_count(&self) -> u64 {
        self.exec_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if hot (should be JIT compiled).
    pub fn is_hot(&self) -> bool {
        self.exec_count() > 1000
    }
}

/// Filter cache for compiled filters.
pub struct FilterCache {
    cache: std::sync::RwLock<HashMap<String, Arc<CompiledFilter>>>,
}

impl FilterCache {
    /// Create a new cache.
    pub fn new() -> Self {
        Self {
            cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Get or compile a filter.
    pub fn get_or_compile(
        &self,
        key: &str,
        build_fn: impl FnOnce() -> IrFunction,
    ) -> Result<Arc<CompiledFilter>, String> {
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(filter) = cache.get(key) {
                return Ok(Arc::clone(filter));
            }
        }

        // Compile new filter
        let function = build_fn();
        let filter = Arc::new(CompiledFilter::compile(function)?);

        // Insert into cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(key.to_string(), Arc::clone(&filter));
        }

        Ok(filter)
    }

    /// Get cached filter count.
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cache.read().unwrap().is_empty()
    }

    /// Clear cache.
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
    }
}

impl Default for FilterCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_filter() {
        // Filter: age >= 18
        let mut builder = IrBuilder::new();
        let age_field = builder.define_field(IrType::I64);
        
        let age = builder.load_i64(age_field);
        let threshold = builder.const_i64(18);
        let result = builder.ge_i64(age, threshold);
        builder.ret(result);

        let func = builder.build();
        let interp = IrInterpreter::new(func).unwrap();

        let mut fields = ArrayFields::new();
        fields.set_i64(0, 21);
        assert!(interp.execute(&fields));

        fields.set_i64(0, 16);
        assert!(!interp.execute(&fields));
    }

    #[test]
    fn test_compound_filter() {
        // Filter: age >= 18 AND score > 50.0
        let mut builder = IrBuilder::new();
        let age_field = builder.define_field(IrType::I64);
        let score_field = builder.define_field(IrType::F64);
        
        let age = builder.load_i64(age_field);
        let age_threshold = builder.const_i64(18);
        let age_ok = builder.ge_i64(age, age_threshold);
        
        let score = builder.load_f64(score_field);
        let score_threshold = builder.const_f64(50.0);
        let score_ok = builder.gt_f64(score, score_threshold);
        
        let result = builder.and(age_ok, score_ok);
        builder.ret(result);

        let func = builder.build();
        let interp = IrInterpreter::new(func).unwrap();

        let mut fields = ArrayFields::new();
        
        // Both pass - age at field 0, score at field 1
        fields.set_i64(age_field as usize, 21);
        fields.set_f64(score_field as usize, 75.0);
        assert!(interp.execute(&fields));

        // Age fails
        fields.set_i64(age_field as usize, 16);
        fields.set_f64(score_field as usize, 75.0);
        assert!(!interp.execute(&fields));

        // Score fails
        fields.set_i64(age_field as usize, 21);
        fields.set_f64(score_field as usize, 30.0);
        assert!(!interp.execute(&fields));
    }

    #[test]
    fn test_or_filter() {
        // Filter: age < 18 OR age > 65
        let mut builder = IrBuilder::new();
        let age_field = builder.define_field(IrType::I64);
        
        let age = builder.load_i64(age_field);
        let low = builder.const_i64(18);
        let high = builder.const_i64(65);
        
        let too_young = builder.lt_i64(age, low);
        let too_old = builder.gt_i64(age, high);
        let result = builder.or(too_young, too_old);
        builder.ret(result);

        let func = builder.build();
        let interp = IrInterpreter::new(func).unwrap();

        let mut fields = ArrayFields::new();
        
        fields.set_i64(0, 10);
        assert!(interp.execute(&fields)); // Too young

        fields.set_i64(0, 70);
        assert!(interp.execute(&fields)); // Too old

        fields.set_i64(0, 30);
        assert!(!interp.execute(&fields)); // In range
    }

    #[test]
    fn test_not_filter() {
        // Filter: NOT (age < 18)
        let mut builder = IrBuilder::new();
        let age_field = builder.define_field(IrType::I64);
        
        let age = builder.load_i64(age_field);
        let threshold = builder.const_i64(18);
        let too_young = builder.lt_i64(age, threshold);
        let result = builder.not(too_young);
        builder.ret(result);

        let func = builder.build();
        let interp = IrInterpreter::new(func).unwrap();

        let mut fields = ArrayFields::new();
        
        fields.set_i64(0, 10);
        assert!(!interp.execute(&fields)); // Too young, NOT fails

        fields.set_i64(0, 21);
        assert!(interp.execute(&fields)); // Not too young, NOT passes
    }

    #[test]
    fn test_validation() {
        // Empty function
        let func = IrFunction::new();
        assert!(func.validate().is_err());

        // Missing Ret
        let mut func = IrFunction::new();
        func.push(IrInst::ConstBool(Reg(0), true));
        assert!(func.validate().is_err());

        // Undefined register
        let mut func = IrFunction::new();
        func.push(IrInst::Ret(Reg(99)));
        assert!(func.validate().is_err());

        // Valid function
        let mut func = IrFunction::new();
        func.alloc_reg(IrType::Bool);
        func.push(IrInst::ConstBool(Reg(0), true));
        func.push(IrInst::Ret(Reg(0)));
        assert!(func.validate().is_ok());
    }

    #[test]
    fn test_compiled_filter() {
        let mut builder = IrBuilder::new();
        let field = builder.define_field(IrType::I64);
        let val = builder.load_i64(field);
        let threshold = builder.const_i64(10);
        let result = builder.ge_i64(val, threshold);
        builder.ret(result);

        let filter = CompiledFilter::compile(builder.build()).unwrap();

        let mut fields = ArrayFields::new();
        fields.set_i64(0, 15);
        
        assert!(filter.execute(&fields));
        assert_eq!(filter.exec_count(), 1);
        
        filter.execute(&fields);
        filter.execute(&fields);
        assert_eq!(filter.exec_count(), 3);
    }

    #[test]
    fn test_filter_cache() {
        let cache = FilterCache::new();

        let filter1 = cache.get_or_compile("age_filter", || {
            let mut builder = IrBuilder::new();
            let field = builder.define_field(IrType::I64);
            let val = builder.load_i64(field);
            let threshold = builder.const_i64(18);
            let result = builder.ge_i64(val, threshold);
            builder.ret(result);
            builder.build()
        }).unwrap();

        let filter2 = cache.get_or_compile("age_filter", || {
            panic!("should not be called - filter should be cached");
        }).unwrap();

        assert!(Arc::ptr_eq(&filter1, &filter2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_ir_type_size() {
        assert_eq!(IrType::Bool.size(), 1);
        assert_eq!(IrType::I64.size(), 8);
        assert_eq!(IrType::F64.size(), 8);
        assert_eq!(IrType::Ptr.size(), 8);
    }

    #[test]
    fn test_instruction_sources() {
        let inst = IrInst::And(Reg(2), Reg(0), Reg(1));
        assert_eq!(inst.sources(), vec![Reg(0), Reg(1)]);
        assert_eq!(inst.dest(), Some(Reg(2)));

        let inst = IrInst::ConstI64(Reg(0), 42);
        assert!(inst.sources().is_empty());
        assert_eq!(inst.dest(), Some(Reg(0)));

        let inst = IrInst::Ret(Reg(0));
        assert_eq!(inst.sources(), vec![Reg(0)]);
        assert_eq!(inst.dest(), None);
    }

    #[test]
    fn test_rt_value_conversions() {
        assert!(RtValue::Bool(true).as_bool());
        assert!(!RtValue::Bool(false).as_bool());
        assert!(RtValue::I64(1).as_bool());
        assert!(!RtValue::I64(0).as_bool());

        assert_eq!(RtValue::I64(42).as_i64(), 42);
        assert_eq!(RtValue::F64(3.14).as_f64(), 3.14);
        assert_eq!(RtValue::I64(10).as_f64(), 10.0);
    }
}

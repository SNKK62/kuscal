use crate::parser::{
    standard_functions, statements_finish, type_check, ExprEnum, Expression, FnDecl, NativeFn,
    Span, Statement, Statements, TypeCheckContext, TypeDecl,
};
use crate::value::{deserialize_size, deserialize_str, serialize_size, serialize_str, Value};
use ruscal::{dprintln, Args, RunMode};
use std::{
    cell::RefCell,
    collections::HashMap,
    error::Error,
    fmt::Display,
    io::{Read, Write},
    rc::Rc,
};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OpCode {
    LoadLiteral,
    Store,
    /// arg0 is target base stack index, pop the first value which is the index, and pop the second value to store
    IndexStore,
    Copy,
    /// arg0 is target base stack index, pop the first value which is the target index
    IndexCopy,
    Dup,
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Call,
    Jmp,
    /// Jump if false
    Jf,
    Not,
    Lt,
    Eq,
    /// Pop n values from the stack where n is given by arg0
    Pop,
    /// Return from function
    Ret,
    // Suspend current function execution where it can resume later
    Yield,
    /// Await a coroutine in progress until the next yield
    Await,
}

macro_rules! impl_op_from {
    ($($op:ident),*) => {
        impl From<u8> for OpCode {
            #[allow(non_upper_case_globals)]
            fn from(o: u8) -> Self {
                $(const $op: u8 = OpCode::$op as u8;)*

                match o {
                    $($op => Self::$op,)*
                    _ => panic!("Opcode \"{:02X}\" unrecognized!", o),
                }
            }
        }
    }
}

impl_op_from!(
    LoadLiteral,
    Store,
    IndexStore,
    Copy,
    IndexCopy,
    Dup,
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Call,
    Jmp,
    Jf,
    Not,
    Lt,
    Eq,
    Pop,
    Ret,
    Yield,
    Await
);

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Instruction {
    op: OpCode,
    arg0: u8,
    // NOTE: for while statement, arg1 is for the stack adustment.
    // in most opcodes, arg1 is not used.
    arg1: u8,
}

impl Instruction {
    pub fn new(op: OpCode, arg0: u8, arg1: u8) -> Self {
        Self { op, arg0, arg1 }
    }

    pub fn serialize(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        writer.write_all(&[self.op as u8, self.arg0, self.arg1])?;
        Ok(())
    }

    pub fn deserialize(reader: &mut impl Read) -> Result<Self, std::io::Error> {
        let mut buf = [0u8; 3];
        reader.read_exact(&mut buf)?;
        Ok(Self::new(buf[0].into(), buf[1], buf[2]))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// Absolute Stack Index
struct StkIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// Instruction Pointer
struct InstPtr(usize); // ip

#[derive(Debug, Clone, Default)]
enum Target {
    #[default]
    Temp,
    // Literal(usize),
    Literal,
    Local(String, Option<TypeDecl>),
}

struct LoopFrame {
    start: StkIdx,
    break_ips: Vec<InstPtr>,
    continue_ips: Vec<(InstPtr, usize)>,
}

impl LoopFrame {
    fn new(start: StkIdx) -> Self {
        Self {
            start,
            break_ips: vec![],
            continue_ips: vec![],
        }
    }
}

#[derive(Debug)]
struct LoopStackUnderflowError;

impl Display for LoopStackUnderflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A break statement outside loop")
    }
}

impl Error for LoopStackUnderflowError {}

struct FnByteCode {
    args: Vec<String>,
    literals: Vec<Value>,
    instructions: Vec<Instruction>,
    cofn: bool,
}

impl FnByteCode {
    fn write_args(args: &[String], writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(args.len(), writer)?;
        for arg in args {
            serialize_str(arg, writer)?;
        }
        Ok(())
    }

    fn write_literals(literals: &[Value], writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(literals.len(), writer)?;
        for lit in literals {
            lit.serialize(writer)?;
        }
        Ok(())
    }

    fn write_insts(instructions: &[Instruction], writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(instructions.len(), writer)?;
        for instruction in instructions {
            instruction.serialize(writer).unwrap();
        }
        Ok(())
    }

    fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        Self::write_args(&self.args, writer)?;
        Self::write_literals(&self.literals, writer)?;
        Self::write_insts(&self.instructions, writer)?;
        writer.write_all(&[self.cofn as u8])?;
        Ok(())
    }

    fn read_args(reader: &mut impl Read) -> std::io::Result<Vec<String>> {
        let num_args = deserialize_size(reader)?;
        let mut args = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            args.push(deserialize_str(reader)?);
        }
        Ok(args)
    }

    fn read_literals(reader: &mut impl Read) -> std::io::Result<Vec<Value>> {
        let num_literals = deserialize_size(reader)?;
        let mut literals = Vec::with_capacity(num_literals);
        for _ in 0..num_literals {
            literals.push(Value::deserialize(reader)?);
        }
        Ok(literals)
    }

    fn read_instructions(reader: &mut impl Read) -> std::io::Result<Vec<Instruction>> {
        let num_instructions = deserialize_size(reader)?;
        let mut instructions = Vec::with_capacity(num_instructions);
        for _ in 0..num_instructions {
            let inst = Instruction::deserialize(reader)?;
            instructions.push(inst);
        }
        Ok(instructions)
    }

    fn deserialize(reader: &mut impl Read) -> std::io::Result<Self> {
        let args = Self::read_args(reader)?;
        let literals = Self::read_literals(reader)?;
        let instructions = Self::read_instructions(reader)?;
        let mut cofn = [0u8];
        reader.read_exact(&mut cofn)?;
        Ok(Self {
            args,
            literals,
            instructions,
            cofn: cofn[0] != 0,
        })
    }

    fn disasm(&self, writer: &mut impl Write) -> std::io::Result<()> {
        disasm_common(&self.literals, &self.instructions, writer)
    }
}

fn disasm_common(
    literals: &[Value],
    instructions: &[Instruction],
    writer: &mut impl Write,
) -> std::io::Result<()> {
    use OpCode::*;
    writeln!(writer, " Literals [{}]", literals.len())?;
    for (i, con) in literals.iter().enumerate() {
        writeln!(writer, "   [{i}] {}", *con)?;
    }

    writeln!(writer, "  Instructions [{}]", instructions.len())?;
    for (i, inst) in instructions.iter().enumerate() {
        match inst.op {
            LoadLiteral => writeln!(
                writer,
                "   [{i}] {:?} {} ({:?})",
                inst.op, inst.arg0, literals[inst.arg0 as usize]
            )?,
            Copy | IndexCopy | Dup | Call | Jmp | Pop | Store | IndexStore | Ret => {
                writeln!(writer, "   [{i}] {:?} {}", inst.op, inst.arg0)?
            }
            Jf => writeln!(writer, "   [{i}] {:?} {} {}", inst.op, inst.arg0, inst.arg1)?,
            _ => writeln!(writer, "   [{i}] {:?}", inst.op)?,
        }
    }
    Ok(())
}

struct Compiler {
    literals: Vec<Value>,
    instructions: Vec<Instruction>,
    target_stack: Vec<Target>,
    funcs: HashMap<String, FnByteCode>,
    loop_stack: Vec<LoopFrame>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            literals: vec![],
            instructions: vec![],
            target_stack: vec![],
            funcs: HashMap::new(),
            loop_stack: vec![],
        }
    }

    fn stack_top(&self) -> StkIdx {
        StkIdx(self.target_stack.len() - 1)
    }

    fn fixup_breaks(&mut self) -> Result<(), Box<dyn Error>> {
        let loop_frame = self.loop_stack.pop().ok_or(LoopStackUnderflowError)?;
        let break_jmp_addr = self.instructions.len();
        for ip in loop_frame.break_ips {
            self.instructions[ip.0].arg0 = break_jmp_addr as u8;
        }
        Ok(())
    }

    fn fixup_continues(&mut self) -> Result<(), Box<dyn Error>> {
        let loop_frame = self.loop_stack.last().ok_or(LoopStackUnderflowError)?;
        let continue_jmp_addr = self.instructions.len();
        for (ip, stk) in &loop_frame.continue_ips {
            self.instructions[ip.0].arg0 = (self.target_stack.len() - stk) as u8;
            self.instructions[ip.0 + 1].arg0 = continue_jmp_addr as u8
        }
        Ok(())
    }

    fn add_literal(&mut self, value: Value) -> u8 {
        let existing = self
            .literals
            .iter()
            .enumerate()
            .find(|(_, val)| **val == value);
        if let Some((i, _)) = existing {
            i as u8
        } else {
            let ret = self.literals.len();
            self.literals.push(value);
            ret as u8
        }
    }

    // return the absolute position of inserted value
    fn add_inst(&mut self, op: OpCode, arg0: u8) -> InstPtr {
        let inst = self.instructions.len();
        self.instructions.push(Instruction { op, arg0, arg1: 0 });
        InstPtr(inst)
    }

    fn add_copy_inst(&mut self, stack_idx: StkIdx) -> InstPtr {
        let inst = self.add_inst(
            OpCode::Copy,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.push(Target::Temp);
        inst
    }

    fn add_index_copy_inst(&mut self, stack_idx: StkIdx) -> InstPtr {
        if self.target_stack.len() < stack_idx.0 + 1 {
            eprintln!("Compiled bytecode so far:");
            disasm_common(&self.literals, &self.instructions, &mut std::io::stderr()).unwrap();
            panic!("Target stack underflow during compilation!");
        }
        let inst = self.add_inst(
            OpCode::IndexCopy,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.pop(); // pop target array index
        self.target_stack.push(Target::Temp); // push value to copy
        inst
    }

    fn add_load_literal_inst(&mut self, lit: u8) -> InstPtr {
        let inst = self.add_inst(OpCode::LoadLiteral, lit);
        // self.target_stack.push(Target::Literal(lit as usize));
        self.target_stack.push(Target::Literal);
        inst
    }

    fn add_binop_inst(&mut self, op: OpCode) -> InstPtr {
        self.target_stack.pop();
        self.add_inst(op, 0)
    }

    fn add_store_inst(&mut self, stack_idx: StkIdx) -> InstPtr {
        if self.target_stack.len() < stack_idx.0 + 1 {
            eprintln!("Compiled bytecode so far:");
            disasm_common(&self.literals, &self.instructions, &mut std::io::stderr()).unwrap();
            panic!("Target stack underflow during compilation!");
        }
        let inst = self.add_inst(
            OpCode::Store,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.pop();
        inst
    }

    fn add_index_store_inst(&mut self, stack_idx: StkIdx) -> InstPtr {
        if self.target_stack.len() < stack_idx.0 + 1 {
            eprintln!("Compiled bytecode so far:");
            disasm_common(&self.literals, &self.instructions, &mut std::io::stderr()).unwrap();
            panic!("Target stack underflow during compilation!");
        }
        let inst = self.add_inst(
            OpCode::IndexStore,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.pop(); // pop array index
        self.target_stack.pop(); // pop value to store
        inst
    }

    fn add_jf_inst(&mut self, coerce_size: Option<u8>) -> InstPtr {
        // push with jump address 0, because it will be set later
        let inst = self.instructions.len();
        if let Some(size) = coerce_size {
            self.instructions.push(Instruction {
                op: OpCode::Jf,
                arg0: 0,
                arg1: size,
            });
        } else {
            self.instructions.push(Instruction {
                op: OpCode::Jf,
                arg0: 0,
                arg1: (self.target_stack.len() - 1) as u8,
            });
        }
        let inst = InstPtr(inst);
        self.target_stack.pop();
        inst
    }

    fn fixup_jmp(&mut self, ip: InstPtr) {
        self.instructions[ip.0].arg0 = self.instructions.len() as u8;
    }

    /// Pop until given stack index
    fn add_pop_until_inst(&mut self, stack_idx: StkIdx) -> Option<InstPtr> {
        if self.target_stack.len() <= stack_idx.0 {
            return None;
        }
        let inst = self.add_inst(
            OpCode::Pop,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.resize(stack_idx.0 + 1, Target::Temp);
        Some(inst)
    }

    fn add_fn(&mut self, name: String, args: &[(Span, TypeDecl)], cofn: bool) {
        self.funcs.insert(
            name,
            FnByteCode {
                args: args.iter().map(|(arg, _)| arg.to_string()).collect(),
                literals: std::mem::take(&mut self.literals),
                instructions: std::mem::take(&mut self.instructions),
                cofn,
            },
        );
    }

    fn write_funcs(&self, writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(self.funcs.len(), writer)?;
        for (name, func) in &self.funcs {
            serialize_str(name, writer)?;
            func.serialize(writer)?;
        }
        Ok(())
    }

    fn compile_expr(&mut self, ex: &Expression) -> Result<StkIdx, Box<dyn Error>> {
        Ok(match &ex.expr {
            ExprEnum::NumLiteral(num) => {
                let id = self.add_literal(Value::F64(*num));
                self.add_load_literal_inst(id);
                self.stack_top()
            }
            ExprEnum::StrLiteral(str) => {
                let id = self.add_literal(Value::Str(str.clone()));
                self.add_load_literal_inst(id);
                self.stack_top()
            }
            ExprEnum::ArrayLiteral(arr) => {
                let mut stk_idx: Option<StkIdx> = None; // INFO: temporary stack index
                arr.iter().for_each(|ex| {
                    let id = self.compile_expr(ex).unwrap();
                    if stk_idx.is_none() {
                        stk_idx = Some(id);
                    };
                });
                if let Some(stk_idx) = stk_idx {
                    stk_idx
                } else {
                    if arr.is_empty() {
                        // INFO: if array is empty, push a 0.0 value temporaryly
                        let id = self.add_literal(Value::F64(0.));
                        self.add_load_literal_inst(id);
                    }
                    self.stack_top()
                }
            }
            ExprEnum::Ident(ident) => {
                let var = self
                    .target_stack
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_i, tgt)| {
                        if let Target::Local(id, _) = tgt {
                            id == ident.fragment()
                        } else {
                            false
                        }
                    });
                if let Some(var) = var {
                    return Ok(StkIdx(var.0));
                } else {
                    return Err(format!("Variable not found: {ident:?}").into());
                }
            }
            ExprEnum::ArrayIndexAccess(ident, indices) => {
                let target_stack = &self.target_stack.clone();
                let var = target_stack.iter().enumerate().rev().find(|(_i, tgt)| {
                    if let Target::Local(id, _) = tgt {
                        id == ident.fragment()
                    } else {
                        false
                    }
                });

                if let Some(var) = var {
                    let (idx, target) = var;
                    let mut ty = match target {
                        Target::Local(_, ty) => ty.as_ref().unwrap(),
                        _ => {
                            return Err(format!("Variable not found: {ident:?}").into());
                        }
                    };
                    let mut i = 0;
                    let mut stk_idxs: Vec<(StkIdx, usize)> = vec![]; // (stack_index, len)
                    loop {
                        match ty {
                            TypeDecl::Array(internal_ty, len) => {
                                let index_ex_idx = self.compile_expr(&indices[i])?;
                                stk_idxs.push((index_ex_idx, *len));
                                i += 1;
                                ty = internal_ty;
                                continue;
                            }
                            TypeDecl::F64 | TypeDecl::Str => {}
                            _ => {
                                return Err(("This type isn't supported.").into());
                            }
                        };
                        break;
                    }

                    let index_idx = self.add_literal(Value::F64(0.)); // temp sum
                    self.add_load_literal_inst(index_idx);
                    let (last_stk_idx, _) = stk_idxs.pop().unwrap();
                    for (stk_idx, len) in stk_idxs.iter() {
                        self.add_copy_inst(*stk_idx);
                        let idx = self.add_literal(Value::F64(*len as f64));
                        self.add_load_literal_inst(idx);
                        self.add_inst(OpCode::Mul, 0);
                        self.target_stack.pop(); // for mul

                        self.add_inst(OpCode::Add, 0);
                        self.target_stack.pop(); // for add
                    }
                    self.add_copy_inst(last_stk_idx);
                    self.add_inst(OpCode::Add, 0);
                    self.target_stack.pop(); // for add

                    self.add_index_copy_inst(StkIdx(idx));
                    self.stack_top()
                } else {
                    return Err(format!("Variable not found: {ident:?}").into());
                }
            }
            ExprEnum::Add(lhs, rhs) => self.bin_op(OpCode::Add, lhs, rhs)?,
            ExprEnum::Sub(lhs, rhs) => self.bin_op(OpCode::Sub, lhs, rhs)?,
            ExprEnum::Mul(lhs, rhs) => self.bin_op(OpCode::Mul, lhs, rhs)?,
            ExprEnum::Div(lhs, rhs) => self.bin_op(OpCode::Div, lhs, rhs)?,
            ExprEnum::And(lhs, rhs) => self.bin_op(OpCode::And, lhs, rhs)?,
            ExprEnum::Or(lhs, rhs) => self.bin_op(OpCode::Or, lhs, rhs)?,
            ExprEnum::Not(ex) => {
                let res = self.compile_expr(ex)?;
                self.add_copy_inst(res);
                self.add_inst(OpCode::Not, 0);
                self.target_stack.pop();
                self.target_stack.push(Target::Temp);
                self.stack_top()
            }
            ExprEnum::Gt(lhs, rhs) => self.bin_op(OpCode::Lt, rhs, lhs)?,
            ExprEnum::Lt(lhs, rhs) => self.bin_op(OpCode::Lt, lhs, rhs)?,
            ExprEnum::Eq(lhs, rhs) => self.bin_op(OpCode::Eq, lhs, rhs)?,
            ExprEnum::Neq(lhs, rhs) => {
                let stk = self.bin_op(OpCode::Eq, lhs, rhs)?;
                self.add_copy_inst(stk);
                self.add_inst(OpCode::Not, 0);
                self.target_stack.pop();
                self.target_stack.push(Target::Temp);
                self.stack_top()
            }
            ExprEnum::FnInvoke(name, args) => {
                let stack_before_args = self.target_stack.len();
                let name = self.add_literal(Value::Str(name.to_string()));
                let args = args
                    .iter()
                    .map(|arg| self.compile_expr(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let stack_before_call = self.target_stack.len();
                self.add_load_literal_inst(name);
                for arg in &args {
                    self.add_copy_inst(*arg);
                }

                self.add_inst(OpCode::Call, args.len() as u8);
                self.target_stack
                    .resize(stack_before_call + 1, Target::Temp);
                self.coerce_stack(StkIdx(stack_before_args));
                self.stack_top()
            }
            ExprEnum::If(cond, true_branch, false_branch) => {
                use OpCode::*;
                let cond = self.compile_expr(cond)?;
                self.add_copy_inst(cond);
                let jf_inst = self.add_jf_inst(None);
                let stack_size_before = self.target_stack.len();
                self.compile_stmts_or_zero(true_branch)?;
                self.coerce_stack(StkIdx(stack_size_before + 1));
                let jmp_inst = self.add_inst(Jmp, 0);
                self.fixup_jmp(jf_inst);
                self.target_stack.resize(stack_size_before, Target::Temp);
                if let Some(false_branch) = false_branch.as_ref() {
                    self.compile_stmts_or_zero(false_branch)?;
                }
                self.coerce_stack(StkIdx(stack_size_before + 1));
                self.fixup_jmp(jmp_inst);
                self.stack_top()
            }
            ExprEnum::Await(ex) => {
                let res = self.compile_expr(ex)?;
                self.add_copy_inst(res);
                self.add_inst(OpCode::Await, 0);
                self.stack_top()
            }
        })
    }

    fn bin_op(
        &mut self,
        op: OpCode,
        lhs: &Expression,
        rhs: &Expression,
    ) -> Result<StkIdx, Box<dyn Error>> {
        let lhs = self.compile_expr(lhs)?;
        let rhs = self.compile_expr(rhs)?;
        self.add_copy_inst(lhs);
        self.add_copy_inst(rhs);
        self.add_inst(op, 0);
        self.target_stack.pop();
        self.target_stack.pop();
        self.target_stack.push(Target::Temp);
        Ok(self.stack_top())
    }

    fn coerce_stack(&mut self, target: StkIdx) {
        match target {
            StkIdx(val) if val < self.target_stack.len() - 1 => {
                self.add_store_inst(target);
                self.add_pop_until_inst(target);
            }
            StkIdx(val) if self.target_stack.len() - 1 < val => {
                for _ in self.target_stack.len() - 1..val {
                    self.add_copy_inst(self.stack_top());
                }
            }
            _ => {}
        }
    }

    fn resolve_array(&mut self, ty: &TypeDecl, ex: &Expression, stk_idx0: &mut Option<StkIdx>) {
        println!("ty: {ty:?}, ex: {ex:?}, stk_idx0: {stk_idx0:?}");
        match ty {
            TypeDecl::Array(ty, len) => match *ty.to_owned() {
                TypeDecl::Array(_, _) => match ex.expr.to_owned() {
                    ExprEnum::ArrayLiteral(ex) => {
                        // TODO: complete the arrays if the assigned length is less than the defined length
                        if ex.len() != *len {
                            panic!("Array length mismatch: expected {len}, got {}", ex.len());
                        }
                        for e in ex {
                            self.resolve_array(ty, &e, stk_idx0);
                        }
                    }
                    _ => {
                        panic!("expression should be array literal");
                    }
                },
                _ => match &ex.expr {
                    ExprEnum::ArrayLiteral(ex) => {
                        let assigned_len = ex.len();
                        if *len < assigned_len {
                            panic!("Array length mismatch: expected {len}, got {assigned_len}");
                        }
                        for e in ex {
                            let stk_idx = self.compile_expr(e);
                            if let Ok(stk_idx) = stk_idx {
                                if stk_idx0.is_none() {
                                    *stk_idx0 = Some(stk_idx);
                                }
                            }
                        }
                        for _ in 0..(*len - assigned_len) {
                            match *ty.to_owned() {
                                TypeDecl::F64 => {
                                    let id = self.add_literal(Value::F64(0.));
                                    self.add_load_literal_inst(id);
                                }
                                TypeDecl::Str => {
                                    let id = self.add_literal(Value::Str("".to_string()));
                                    self.add_load_literal_inst(id);
                                }
                                _ => panic!("Unsupported type"),
                            }
                        }
                    }
                    _ => {
                        panic!("expression should be array literal");
                    }
                },
            },
            _ => panic!("This type should be array"),
        }
    }

    fn compile_stmts(&mut self, stmts: &Statements) -> Result<Option<StkIdx>, Box<dyn Error>> {
        let mut last_result = None;
        for stmt in stmts {
            match stmt {
                Statement::Expression(ex) => {
                    last_result = Some(self.compile_expr(ex)?);
                }
                Statement::VarDef { name, ex, td, .. } => match td {
                    TypeDecl::Array(_, _) => {
                        let mut stk_idx = None;
                        self.resolve_array(td, ex, &mut stk_idx);

                        let mut tmp_ty = td;
                        let mut sum_len = 1;
                        while let TypeDecl::Array(ty, len) = tmp_ty {
                            sum_len *= len;
                            tmp_ty = ty.as_ref();
                        }

                        for i in 0..sum_len {
                            self.add_copy_inst(StkIdx(stk_idx.unwrap().0 + i));
                        }

                        if let Some(stk_idx) = stk_idx {
                            self.target_stack[stk_idx.0 + sum_len] =
                                Target::Local(name.to_string(), Some(td.clone()));
                        } else {
                            panic!("Array index not found");
                        }
                    }
                    _ => {
                        let mut stk_idx = self.compile_expr(ex)?;
                        if !matches!(self.target_stack[stk_idx.0], Target::Temp) {
                            self.add_copy_inst(stk_idx);
                            stk_idx = self.stack_top();
                        }
                        self.target_stack[stk_idx.0] =
                            Target::Local(name.to_string(), Some(td.clone()));
                    }
                },
                Statement::VarAssign { name, ex, .. } => {
                    let stk_ex = self.compile_expr(ex)?;
                    let (stk_local, _) = self
                        .target_stack
                        .iter_mut()
                        .enumerate()
                        .rev()
                        .find(|(_, tgt)| {
                            if let Target::Local(tgt, _) = tgt {
                                tgt == name.fragment()
                            } else {
                                false
                            }
                        })
                        .ok_or_else(|| format!("Variable name not found: {name}"))?;
                    self.add_copy_inst(stk_ex);
                    self.add_store_inst(StkIdx(stk_local));
                }
                Statement::ArrayIndexAssign {
                    name, indices, ex, ..
                } => {
                    let stk_ex = self.compile_expr(ex)?;
                    let mut target_stack = self.target_stack.clone();
                    let opt = target_stack
                        .iter_mut()
                        .enumerate()
                        .rev()
                        .find(|(_, tgt)| {
                            if let Target::Local(tgt, _) = tgt {
                                tgt == name.fragment()
                            } else {
                                false
                            }
                        })
                        .ok_or_else(|| format!("Variable name not found: {name}"))?;
                    let (stk_local, target) = opt;

                    let mut ty = match target {
                        Target::Local(_, ty) => ty.as_ref().unwrap(),
                        _ => {
                            return Err(("target should be Target::Local").into());
                        }
                    };
                    let mut i = 0;
                    let mut stk_idxs: Vec<(StkIdx, usize)> = vec![]; // (stack_index, len)
                    loop {
                        match ty {
                            TypeDecl::Array(internal_ty, len) => {
                                let ex = &indices[i];
                                let index_ex_idx = self.compile_expr(ex)?;
                                stk_idxs.push((index_ex_idx, *len));
                                i += 1;
                                ty = internal_ty;
                                continue;
                            }
                            TypeDecl::F64 | TypeDecl::Str => {}
                            _ => {
                                return Err(("This type isn't supported.").into());
                            }
                        };
                        break;
                    }

                    let index_idx = self.add_literal(Value::F64(0.)); // temp sum
                    self.add_load_literal_inst(index_idx);
                    let (last_stk_idx, _) = stk_idxs.pop().unwrap();
                    for (stk_idx, len) in stk_idxs.iter() {
                        self.add_copy_inst(*stk_idx);
                        let idx = self.add_literal(Value::F64(*len as f64));
                        self.add_load_literal_inst(idx);
                        self.add_inst(OpCode::Mul, 0);
                        self.target_stack.pop(); // for mul

                        self.add_inst(OpCode::Add, 0);
                        self.target_stack.pop(); // for add
                    }
                    self.add_copy_inst(last_stk_idx);
                    self.add_inst(OpCode::Add, 0);
                    self.target_stack.pop(); // for add

                    let index_stk_idx = self.stack_top();

                    self.add_copy_inst(stk_ex);
                    self.add_copy_inst(index_stk_idx);
                    self.add_index_store_inst(StkIdx(stk_local));
                }
                Statement::For {
                    loop_var,
                    start,
                    end,
                    stmts,
                    ..
                } => {
                    let stk_start = self.compile_expr(start)?;
                    let stk_end = self.compile_expr(end)?;
                    dprintln!("start: {stk_start:?} end: {stk_end:?}");
                    self.add_copy_inst(stk_start);
                    let stk_loop_var = self.stack_top();
                    self.target_stack[stk_loop_var.0] =
                        Target::Local(loop_var.to_string(), Some(TypeDecl::I64));
                    dprintln!("after start: {:?}", self.target_stack);
                    let inst_check_exit = self.instructions.len();
                    self.add_copy_inst(stk_loop_var);
                    self.add_copy_inst(stk_end);
                    dprintln!("before cmp: {:?}", self.target_stack);
                    self.add_binop_inst(OpCode::Lt);
                    let jf_inst = self.add_jf_inst(None);
                    dprintln!("start in loop: {:?}", self.target_stack);
                    self.loop_stack.push(LoopFrame::new(stk_loop_var));
                    self.compile_stmts(stmts)?;
                    self.fixup_continues()?;
                    let one = self.add_literal(Value::F64(1.0));
                    dprintln!("end in loop: {:?}", self.target_stack);
                    self.add_copy_inst(stk_loop_var);
                    self.add_load_literal_inst(one);
                    self.add_inst(OpCode::Add, 0);
                    self.target_stack.pop();
                    self.add_store_inst(stk_loop_var);
                    self.add_pop_until_inst(stk_loop_var);
                    self.add_inst(OpCode::Jmp, inst_check_exit as u8);
                    self.fixup_jmp(jf_inst);
                    self.fixup_breaks()?;
                }
                Statement::While { cond, stmts, .. } => {
                    let inst_check_exit = self.instructions.len();
                    let stk_before_cond = self.stack_top();

                    self.compile_expr(cond)?;
                    let jf_inst = self.add_jf_inst(Some((stk_before_cond.0 + 1) as u8));
                    dprintln!("start in loop: {:?}", self.target_stack);

                    self.loop_stack.push(LoopFrame::new(stk_before_cond));

                    self.compile_stmts(stmts)?;
                    self.fixup_continues()?;

                    self.add_pop_until_inst(stk_before_cond);
                    self.add_inst(OpCode::Jmp, inst_check_exit as u8);
                    self.fixup_jmp(jf_inst);
                    self.fixup_breaks()?;
                }
                Statement::Break => {
                    let start = self
                        .loop_stack
                        .last()
                        .map(|loop_frame| loop_frame.start)
                        .ok_or(LoopStackUnderflowError)?;
                    self.add_pop_until_inst(start);

                    let loop_frame = self.loop_stack.last_mut().ok_or(LoopStackUnderflowError)?;
                    let break_ip = self.instructions.len();
                    loop_frame.break_ips.push(InstPtr(break_ip));
                    self.add_inst(OpCode::Jmp, 0);
                }
                Statement::Continue => {
                    let start = self
                        .loop_stack
                        .last()
                        .map(|frame| frame.start)
                        .ok_or(LoopStackUnderflowError)?;
                    self.add_pop_until_inst(start);

                    let loop_frame = self.loop_stack.last_mut().ok_or(LoopStackUnderflowError)?;
                    let continue_ip = self.instructions.len();
                    loop_frame
                        .continue_ips
                        .push((InstPtr(continue_ip), self.target_stack.len()));
                    self.add_inst(OpCode::Dup, 0);
                    self.add_inst(OpCode::Jmp, 0);
                }
                Statement::FnDef {
                    name,
                    args,
                    stmts,
                    cofn,
                    ..
                } => {
                    let literals = std::mem::take(&mut self.literals);
                    let instructions = std::mem::take(&mut self.instructions);
                    let target_stack = std::mem::take(&mut self.target_stack);
                    self.target_stack = args
                        .iter()
                        .map(|arg| {
                            let ty = (arg.1).clone();
                            Target::Local(arg.0.to_string(), Some(ty))
                        })
                        .collect();
                    self.compile_stmts(stmts)?;
                    self.add_fn(name.to_string(), args, *cofn);
                    self.literals = literals;
                    self.instructions = instructions;
                    self.target_stack = target_stack;
                }
                Statement::Return(ex) => {
                    let res = self.compile_expr(ex)?;
                    self.add_copy_inst(res);
                    self.add_inst(OpCode::Ret, (self.target_stack.len() - res.0 - 1) as u8);
                }
                Statement::Yield(ex) => {
                    let res = self.compile_expr(ex)?;
                    self.add_inst(OpCode::Yield, (self.target_stack.len() - res.0 - 1) as u8);
                    self.target_stack.pop();
                }
            }
        }
        Ok(last_result)
    }

    fn compile_stmts_or_zero(&mut self, stmts: &Statements) -> Result<StkIdx, Box<dyn Error>> {
        Ok(self.compile_stmts(stmts)?.unwrap_or_else(|| {
            let id = self.add_literal(Value::F64(0.));
            self.add_load_literal_inst(id);
            self.stack_top()
        }))
    }

    fn compile(&mut self, stmts: &Statements) -> Result<(), Box<dyn std::error::Error>> {
        let name = "main";
        self.compile_stmts_or_zero(stmts)?;
        self.add_fn(name.to_string(), &[], false);
        Ok(())
    }

    fn disasm(&self, writer: &mut impl Write) -> std::io::Result<()> {
        for (name, fn_def) in &self.funcs {
            if fn_def.cofn {
                writeln!(writer, "Coroutine {name:?}:")?;
            } else {
                writeln!(writer, "Function {name:?}:")?;
            }
            fn_def.disasm(writer)?;
        }
        Ok(())
    }
}

fn write_program(
    source_file: &str,
    source: &str,
    writer: &mut impl Write,
    out_file: &str,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut compiler = Compiler::new();
    let stmts = statements_finish(Span::new(source)).map_err(|e| {
        format!(
            "{}:{}:{}: {}",
            source_file,
            e.input.location_line(),
            e.input.get_utf8_column(),
            e
        )
    })?;

    if args.show_ast {
        // dprintln!("AST: {stmts:#?}");
        println!("AST: {stmts:#?}");
    }

    match type_check(&stmts, &mut TypeCheckContext::new()) {
        Ok(_) => println!("Typecheck Ok"),
        Err(e) => {
            return Err(format!(
                "{}:{}:{}: {}",
                source_file,
                e.span.location_line(),
                e.span.get_utf8_column(),
                e
            )
            .into())
        }
    }

    if matches!(args.run_mode, RunMode::TypeCheck) {
        return Ok(());
    }

    compiler.compile(&stmts)?;

    if args.disasm {
        compiler.disasm(&mut std::io::stdout())?;
    }

    compiler.write_funcs(writer)?;
    println!(
        "Writeen {} literals and {} instructions to {out_file:?}",
        compiler.literals.len(),
        compiler.instructions.len()
    );
    Ok(())
}

enum FnDef {
    User(Rc<FnByteCode>),
    Native(NativeFn<'static>),
}

pub struct ByteCode {
    funcs: HashMap<String, FnDef>,
}

impl ByteCode {
    fn new() -> Self {
        Self {
            funcs: HashMap::new(),
        }
    }

    fn read_funcs(&mut self, reader: &mut impl Read) -> std::io::Result<()> {
        let num_funcs = deserialize_size(reader)?;
        let mut funcs: HashMap<_, _> = standard_functions()
            .into_iter()
            .filter_map(|(name, f)| {
                if let FnDecl::Native(f) = f {
                    Some((name, FnDef::Native(f)))
                } else {
                    None
                }
            })
            .collect();
        for _ in 0..num_funcs {
            let name = deserialize_str(reader)?;
            funcs.insert(name, FnDef::User(Rc::new(FnByteCode::deserialize(reader)?)));
        }
        self.funcs = funcs;
        Ok(())
    }
}

pub enum YieldResult {
    Finished(Value),
    Suspend(Value),
}

struct StackFrame {
    fn_def: Rc<FnByteCode>,
    args: usize,
    stack: Vec<Value>,
    ip: usize,
}

impl StackFrame {
    fn new(fn_def: Rc<FnByteCode>, args: Vec<Value>) -> Self {
        Self {
            fn_def,
            args: args.len(),
            stack: args,
            ip: 0,
        }
    }

    fn inst(&self) -> Option<Instruction> {
        let ret = self.fn_def.instructions.get(self.ip)?;
        dprintln!("interpret[{}]: {:?} stack: {:?}", self.ip, ret, self.stack);
        Some(*ret)
    }
}

pub struct Vm {
    bytecode: Rc<ByteCode>,
    stack_frames: Vec<StackFrame>,
}

impl std::fmt::Debug for Vm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Vm>")
    }
}

impl Vm {
    pub fn new(bytecode: Rc<ByteCode>) -> Self {
        Self {
            bytecode,
            stack_frames: vec![],
        }
    }

    fn top(&self) -> Result<&StackFrame, String> {
        self.stack_frames
            .last()
            .ok_or_else(|| "Stack frame underflow".to_string())
    }

    fn top_mut(&mut self) -> Result<&mut StackFrame, String> {
        self.stack_frames
            .last_mut()
            .ok_or_else(|| "Stack frame underflow".to_string())
    }

    #[allow(dead_code)]
    fn run_fn(
        &mut self,
        fn_name: &str,
        args: &[Value],
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let fn_def = self
            .bytecode
            .funcs
            .get(fn_name)
            .ok_or_else(|| format!("Function {fn_name:?} was not found"))?;
        let fn_def = match fn_def {
            FnDef::User(user) => user.clone(),
            FnDef::Native(n) => return Ok((*n.code)(args)),
        };
        self.stack_frames
            .push(StackFrame::new(fn_def, args.to_vec()));

        match self.interpret()? {
            YieldResult::Finished(val) => Ok(val),
            YieldResult::Suspend(_) => Err("Yielded at toplevel".into()),
        }
    }

    pub fn init_fn(
        &mut self,
        fn_name: &str,
        args: &[Value],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fn_def = self
            .bytecode
            .funcs
            .get(fn_name)
            .ok_or_else(|| format!("Function {fn_name:?} was not found"))?;
        let fn_def = match fn_def {
            FnDef::User(user) => user.clone(),
            FnDef::Native(_) => {
                return Err(
                    "Native function cannot be called as a coroutine. Use `run_fn` instead.".into(),
                )
            }
        };
        self.stack_frames
            .push(StackFrame::new(fn_def, args.to_vec()));
        Ok(())
    }

    fn return_fn(&mut self, stack_pos: u8) -> Result<Option<YieldResult>, Box<dyn Error>> {
        let top_frame = self
            .stack_frames
            .pop()
            .ok_or_else(|| "Stack frame underflow".to_string())?;
        let res = top_frame
            .stack
            .get(top_frame.stack.len() - stack_pos as usize - 1)
            .ok_or_else(|| "Stack underflow at last".to_string())?
            .clone();
        let args = top_frame.args;

        if self.stack_frames.is_empty() {
            return Ok(Some(YieldResult::Finished(res)));
        }

        dprintln!("Return {}", res);

        let stack = &mut self.top_mut()?.stack;
        stack.resize(stack.len() - args - 1, Value::F64(0.));
        stack.push(res);
        self.top_mut()?.ip += 1;
        Ok(None)
    }

    pub fn interpret(&mut self) -> Result<YieldResult, Box<dyn std::error::Error>> {
        loop {
            let instruction = if let Some(instruction) = self.top()?.inst() {
                instruction
            } else {
                if let Some(res) = self.return_fn(0)? {
                    return Ok(res);
                }
                continue;
            };
            // debug instruction
            // println!("- instruction: {:?}", instruction,);

            match instruction.op {
                OpCode::LoadLiteral => {
                    let stack_frame = self.top_mut()?;
                    stack_frame
                        .stack
                        .push(stack_frame.fn_def.literals[instruction.arg0 as usize].clone())
                }
                OpCode::Store => {
                    let stack = &mut self.top_mut()?.stack;
                    let idx = stack.len() - instruction.arg0 as usize - 1;
                    let value = stack.pop().expect("Store needs an argument");
                    stack[idx] = value;
                }
                OpCode::IndexStore => {
                    let stack = &mut self.top_mut()?.stack;
                    let stack_length = stack.len();
                    let array_index = stack.pop().expect("IndexStore needs an array index");
                    // TODO: index should be i64 (or usize)
                    let array_index = if let Value::F64(index) = array_index {
                        index as usize
                    } else if let Value::I64(index) = array_index {
                        index as usize
                    } else {
                        return Err("IndexStore needs an array index".into());
                    };
                    let idx = stack_length - instruction.arg0 as usize + array_index - 1;
                    let value = stack.pop().expect("IndexStore needs an argument");
                    stack[idx] = value;
                }
                OpCode::Copy => {
                    let stack = &mut self.top_mut()?.stack;
                    stack.push(stack[stack.len() - instruction.arg0 as usize - 1].clone());
                }
                OpCode::IndexCopy => {
                    let stack = &mut self.top_mut()?.stack;
                    let stack_length = stack.len();
                    let array_index = stack.pop().expect("IndexCopy needs an array index");
                    // TODO: index should be i64 (or usize)
                    let array_index = if let Value::F64(index) = array_index {
                        index as usize
                    } else if let Value::I64(index) = array_index {
                        index as usize
                    } else {
                        return Err("IndexCopy needs an array index".into());
                    };
                    let idx = stack_length - instruction.arg0 as usize + array_index - 1;
                    stack.push(stack[idx].clone());
                }
                OpCode::Dup => {
                    let stack = &mut self.top_mut()?.stack;
                    let top = stack.last().unwrap().clone();
                    stack.extend((0..instruction.arg0).map(|_| top.clone()));
                }
                OpCode::Add => Self::interpret_bin_op_str(
                    &mut self.top_mut()?.stack,
                    |lhs, rhs| lhs + rhs,
                    |lhs, rhs| lhs + rhs,
                    |lhs, rhs| Some(format!("{lhs}{rhs}")),
                ),
                OpCode::Sub => Self::interpret_bin_op(
                    &mut self.top_mut()?.stack,
                    |lhs, rhs| lhs - rhs,
                    |lhs, rhs| lhs - rhs,
                ),
                OpCode::Mul => Self::interpret_bin_op(
                    &mut self.top_mut()?.stack,
                    |lhs, rhs| lhs * rhs,
                    |lhs, rhs| lhs * rhs,
                ),
                OpCode::Div => Self::interpret_bin_op(
                    &mut self.top_mut()?.stack,
                    |lhs, rhs| lhs / rhs,
                    |lhs, rhs| lhs / rhs,
                ),
                OpCode::And => Self::interpret_bin_op(
                    &mut self.top_mut()?.stack,
                    |lhs, rhs| (lhs != 0. && rhs != 0.) as i32 as f64,
                    |lhs, rhs| (lhs != 0 && rhs != 0) as i64,
                ),
                OpCode::Or => Self::interpret_bin_op(
                    &mut self.top_mut()?.stack,
                    |lhs, rhs| (lhs != 0. || rhs != 0.) as i32 as f64,
                    |lhs, rhs| (lhs != 0 || rhs != 0) as i64,
                ),
                OpCode::Call => {
                    let stack = &self.top()?.stack;
                    let args = &stack[stack.len() - instruction.arg0 as usize..];
                    let fname = &stack[stack.len() - instruction.arg0 as usize - 1];
                    let Value::Str(fname) = fname else {
                        panic!(
                            "Function name shall be a string: {fname:?} in fn {:?}",
                            self.top()?.stack
                        );
                    };
                    let fn_def = self
                        .bytecode
                        .funcs
                        .get(fname)
                        .ok_or_else(|| format!("Function name shall be a string: {fname:?}"))?;
                    match fn_def {
                        FnDef::User(user_fn) => {
                            if user_fn.cofn {
                                let mut vm = Vm::new(self.bytecode.clone());
                                vm.stack_frames
                                    .push(StackFrame::new(user_fn.clone(), args.to_vec()));
                                let stack = &mut self.top_mut()?.stack;
                                stack.resize(
                                    stack.len() - instruction.arg0 as usize - 1,
                                    Value::F64(0.),
                                );
                                stack.push(Value::Coro(Rc::new(RefCell::new(vm))));
                            } else {
                                self.stack_frames
                                    .push(StackFrame::new(user_fn.clone(), args.to_vec()));
                                continue;
                            }
                        }
                        FnDef::Native(native) => {
                            let res = (native.code)(args);
                            let stack = &mut (self.top_mut()?.stack);
                            stack.resize(
                                stack.len() - instruction.arg0 as usize - 1,
                                Value::F64(0.),
                            );
                            stack.push(res);
                        }
                    }
                }
                OpCode::Jmp => {
                    self.top_mut()?.ip = instruction.arg0 as usize;
                    continue;
                }
                OpCode::Jf => {
                    let stack = &mut self.top_mut()?.stack;
                    let cond = stack.pop().expect("Jf needs an argument");
                    if cond.coerce_f64() == 0. {
                        self.top_mut()?.ip = instruction.arg0 as usize;
                        self.top_mut()?
                            .stack
                            .resize(instruction.arg1 as usize, Value::F64(0.));
                        continue;
                    }
                }
                OpCode::Not => {
                    let stack = &mut self.top_mut()?.stack;
                    let top = stack.last().unwrap().clone();
                    stack.pop();
                    stack.push(Value::F64(if top.coerce_f64() == 0. { 1. } else { 0. }));
                }
                OpCode::Lt => Self::interpret_bin_op(
                    &mut self.top_mut()?.stack,
                    |lhs, rhs| (lhs < rhs) as i32 as f64,
                    |lhs, rhs| (lhs < rhs) as i64,
                ),
                OpCode::Eq => Self::interpret_bin_op(
                    &mut self.top_mut()?.stack,
                    |lhs, rhs| (lhs == rhs) as i32 as f64,
                    |lhs, rhs| (lhs == rhs) as i64,
                ),
                OpCode::Pop => {
                    let stack = &mut self.top_mut()?.stack;
                    stack.resize(stack.len() - instruction.arg0 as usize, Value::default());
                }
                OpCode::Ret => {
                    if let Some(res) = self.return_fn(instruction.arg0)? {
                        return Ok(res);
                    }
                    continue;
                }
                OpCode::Yield => {
                    let top_frame = self.top_mut()?;
                    let res = top_frame
                        .stack
                        .pop()
                        .ok_or_else(|| "Stack underflow".to_string())?;
                    top_frame.ip += 1;
                    return Ok(YieldResult::Suspend(res));
                }
                OpCode::Await => {
                    let vms = self
                        .top_mut()?
                        .stack
                        .pop()
                        .ok_or_else(|| "Stack underflow".to_string())?;
                    let Value::Coro(vm) = vms else {
                        return Err("Await keyword applied to a non-coroutine".into());
                    };
                    match vm.borrow_mut().interpret() {
                        Ok(YieldResult::Finished(_)) => (),
                        Ok(YieldResult::Suspend(value)) => {
                            self.top_mut()?.stack.push(value);
                        }
                        Err(e) => {
                            eprintln!("Runtime error: {e:?}");
                        }
                    };
                }
            }
            // debug stack
            // println!("stack: {:?}", self.top()?.stack);
            self.top_mut()?.ip += 1;
        }
    }

    fn interpret_bin_op_str(
        stack: &mut Vec<Value>,
        op_f64: impl FnOnce(f64, f64) -> f64,
        op_i64: impl FnOnce(i64, i64) -> i64,
        op_str: impl FnOnce(&str, &str) -> Option<String>,
    ) {
        use Value::*;
        let rhs = stack.pop().expect("Stack underflow");
        let lhs = stack.pop().expect("Stack underflow");
        let res = match (lhs, rhs) {
            (F64(lhs), F64(rhs)) => F64(op_f64(lhs, rhs)),
            (I64(lhs), I64(rhs)) => I64(op_i64(lhs, rhs)),
            (F64(lhs), I64(rhs)) => F64(op_f64(lhs, rhs as f64)),
            (I64(lhs), F64(rhs)) => F64(op_f64(lhs as f64, rhs)),
            (Str(lhs), Str(rhs)) => {
                if let Some(res) = op_str(&lhs, &rhs) {
                    Str(res)
                } else {
                    panic!("Operation not supported for strings: {lhs:?} {rhs:?}");
                }
            }
            (lhs, rhs) => panic!("Operation not supported: {lhs:?} {rhs:?}"),
        };
        stack.push(res);
    }

    fn interpret_bin_op(
        stack: &mut Vec<Value>,
        op_f64: impl FnOnce(f64, f64) -> f64,
        op_i64: impl FnOnce(i64, i64) -> i64,
    ) {
        Self::interpret_bin_op_str(stack, op_f64, op_i64, |_, _| None);
    }

    fn back_trace(&self) {
        for (i, frame) in self.stack_frames.iter().rev().enumerate() {
            println!("[{}]: {:?}", i, frame.stack);
        }
    }
}

pub fn compile(
    writer: &mut impl Write,
    args: &Args,
    out_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let src = args.source.as_ref().ok_or_else(|| {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Please specify source file to compile after -c".to_string(),
        ))
    })?;
    let source = std::fs::read_to_string(src)?;
    write_program(src, &source, writer, out_file, args)
}

pub fn read_program(reader: &mut impl Read) -> std::io::Result<ByteCode> {
    let mut bytecode = ByteCode::new();
    bytecode.read_funcs(reader)?;
    Ok(bytecode)
}

pub fn debugger(vm: &Vm) -> bool {
    println!("[c]ontinue/[p]rint/[e]xit/[bt]race?");
    loop {
        let mut buffer = String::new();
        if std::io::stdin().read_line(&mut buffer).is_ok() {
            match buffer.trim() {
                "c" => return false,
                "p" => {
                    println!("Stack: {:?}", vm.top().unwrap().stack);
                }
                "e" => return true,
                "bt" => vm.back_trace(),
                _ => println!("Please say [c]ontinue/[p]rint/[b]reak/[bt]race"),
            }
        }
    }
}

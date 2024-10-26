use std::{
    error::Error,
    fmt::Display,
    io::{BufReader, BufWriter, Read, Write},
};
mod simple_parser;
use ruscal::{dprintln, parse_args, Args, RunMode};
use simple_parser::{statements_finish, Expression, Statement, Statements};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OpCode {
    LoadLiteral,
    Store,
    Copy,
    Dup,
    Add,
    Sub,
    Mul,
    Div,
    Call,
    Jmp,
    Jf, // Jump if false
    Lt,
    Pop, // Pop n values from the stack where n is given by arg0
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
    Copy,
    Dup,
    Add,
    Sub,
    Mul,
    Div,
    Call,
    Jmp,
    Jf,
    Lt,
    Pop
);

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Instruction {
    op: OpCode,
    arg0: u8,
}

impl Instruction {
    fn new(op: OpCode, arg0: u8) -> Self {
        Self { op, arg0 }
    }

    fn serialize(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        writer.write_all(&[self.op as u8, self.arg0])?;
        Ok(())
    }

    fn deserialize(reader: &mut impl Read) -> Result<Self, std::io::Error> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(Self::new(buf[0].into(), buf[1]))
    }
}

fn serialize_size(sz: usize, writer: &mut impl Write) -> std::io::Result<()> {
    writer.write_all(&(sz as u32).to_le_bytes())
}

fn deserialize_size(reader: &mut impl Read) -> std::io::Result<usize> {
    let mut buf = [0u8; std::mem::size_of::<u32>()];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf) as usize)
}

fn serialize_str(s: &str, writer: &mut impl Write) -> std::io::Result<()> {
    serialize_size(s.len(), writer)?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}

fn deserialize_str(reader: &mut impl Read) -> std::io::Result<String> {
    let mut buf = vec![0u8; deserialize_size(reader)?];
    reader.read_exact(&mut buf)?;
    let s = String::from_utf8(buf).unwrap();
    Ok(s)
}

#[repr(u8)]
enum ValueKind {
    F64,
    Str,
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    F64(f64),
    Str(String),
}

impl Default for Value {
    fn default() -> Self {
        Self::F64(0.)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F64(value) => write!(f, "{value}"),
            Self::Str(value) => write!(f, "{value:?}"),
        }
    }
}

impl Value {
    fn kind(&self) -> ValueKind {
        match self {
            Self::F64(_) => ValueKind::F64,
            Self::Str(_) => ValueKind::Str,
        }
    }

    fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let kind = self.kind() as u8;
        writer.write_all(&[kind])?;
        match self {
            Self::F64(value) => {
                writer.write_all(&value.to_le_bytes())?;
            }
            Self::Str(value) => serialize_str(value, writer)?,
        };
        Ok(())
    }

    #[allow(non_upper_case_globals)]
    fn deserialize(reader: &mut impl Read) -> std::io::Result<Self> {
        const F64: u8 = ValueKind::F64 as u8;
        const Str: u8 = ValueKind::Str as u8;

        let mut kind_buf = [0u8; 1];
        reader.read_exact(&mut kind_buf)?;
        match kind_buf[0] {
            F64 => {
                let mut buf = [0u8; std::mem::size_of::<f64>()];
                reader.read_exact(&mut buf)?;
                Ok(Value::F64(f64::from_le_bytes(buf)))
            }
            Str => Ok(Value::Str(deserialize_str(reader)?)),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "ValueKind {} does not match to any known kinds",
                    kind_buf[0]
                ),
            )),
        }
    }

    fn coerce_f64(&self) -> f64 {
        match self {
            Self::F64(value) => *value,
            _ => panic!("Coercion failed: {:?} cannot be coerced to f64", self),
        }
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
    Literal(usize),
    Local(String),
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

struct Compiler {
    literals: Vec<Value>,
    instructions: Vec<Instruction>,
    target_stack: Vec<Target>,
    loop_stack: Vec<LoopFrame>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            literals: vec![],
            instructions: vec![],
            target_stack: vec![],
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
        self.instructions.push(Instruction { op, arg0 });
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

    fn add_load_literal_inst(&mut self, lit: u8) -> InstPtr {
        let inst = self.add_inst(OpCode::LoadLiteral, lit);
        self.target_stack.push(Target::Literal(lit as usize));
        inst
    }

    fn add_binop_inst(&mut self, op: OpCode) -> InstPtr {
        self.target_stack.pop();
        self.add_inst(op, 0)
    }

    fn add_store_inst(&mut self, stack_idx: StkIdx) -> InstPtr {
        let inst = self.add_inst(
            OpCode::Store,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.pop();
        inst
    }

    fn add_jf_inst(&mut self) -> InstPtr {
        // push with jump address 0, because it will be set later
        let inst = self.add_inst(OpCode::Jf, 0);
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

    fn write_literals(&self, writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(self.literals.len(), writer)?;
        for value in &self.literals {
            value.serialize(writer)?;
        }
        Ok(())
    }

    fn write_insts(&self, writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(self.instructions.len(), writer)?;
        for instruction in &self.instructions {
            instruction.serialize(writer).unwrap();
        }
        Ok(())
    }

    fn compile_expr(&mut self, ex: &Expression) -> Result<StkIdx, Box<dyn Error>> {
        Ok(match ex {
            Expression::NumLiteral(num) => {
                let id = self.add_literal(Value::F64(*num));
                self.add_load_literal_inst(id);
                self.stack_top()
            }
            Expression::Ident(ident) => {
                let var = self.target_stack.iter().enumerate().find(|(_i, tgt)| {
                    if let Target::Local(id) = tgt {
                        id == ident
                    } else {
                        false
                    }
                });
                if let Some(var) = var {
                    return Ok(StkIdx(var.0));
                } else {
                    dprintln!("stack: {:?}", self.target_stack);
                    return Err(format!("Variable not found: {ident:?}").into());
                }
            }
            Expression::Add(lhs, rhs) => self.bin_op(OpCode::Add, lhs, rhs)?,
            Expression::Sub(lhs, rhs) => self.bin_op(OpCode::Sub, lhs, rhs)?,
            Expression::Mul(lhs, rhs) => self.bin_op(OpCode::Mul, lhs, rhs)?,
            Expression::Div(lhs, rhs) => self.bin_op(OpCode::Div, lhs, rhs)?,
            Expression::Gt(lhs, rhs) => self.bin_op(OpCode::Lt, rhs, lhs)?,
            Expression::Lt(lhs, rhs) => self.bin_op(OpCode::Lt, lhs, rhs)?,
            Expression::FnInvoke(name, args) => {
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
            Expression::If(cond, true_branch, false_branch) => {
                use OpCode::*;
                let cond = self.compile_expr(cond)?;
                self.add_copy_inst(cond);
                let jf_inst = self.add_jf_inst();
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

    fn compile_stmts(&mut self, stmts: &Statements) -> Result<Option<StkIdx>, Box<dyn Error>> {
        let mut last_result = None;
        for stmt in stmts {
            match stmt {
                Statement::Expression(ex) => {
                    last_result = Some(self.compile_expr(ex)?);
                }
                Statement::VarDef(vname, ex) => {
                    let mut ex = self.compile_expr(ex)?;
                    if matches!(self.target_stack[ex.0], Target::Local(_)) {
                        self.add_copy_inst(ex);
                        ex = self.stack_top();
                    }
                    self.target_stack[ex.0] = Target::Local(vname.to_string());
                }
                Statement::VarAssign(vname, ex) => {
                    let stk_ex = self.compile_expr(ex)?;
                    let (stk_local, _) = self
                        .target_stack
                        .iter_mut()
                        .enumerate()
                        .find(|(_, tgt)| {
                            if let Target::Local(tgt) = tgt {
                                tgt == vname
                            } else {
                                false
                            }
                        })
                        .ok_or_else(|| format!("Variable name not found: {vname}"))?;
                    self.add_copy_inst(stk_ex);
                    self.add_store_inst(StkIdx(stk_local));
                }
                Statement::For {
                    loop_var,
                    start,
                    end,
                    stmts,
                } => {
                    let stk_start = self.compile_expr(start)?;
                    let stk_end = self.compile_expr(end)?;
                    dprintln!("start: {stk_start:?} end: {stk_end:?}");
                    self.add_copy_inst(stk_start);
                    let stk_loop_var = self.stack_top();
                    self.target_stack[stk_loop_var.0] = Target::Local(loop_var.to_string());
                    dprintln!("after start: {:?}", self.target_stack);
                    let inst_check_exit = self.instructions.len();
                    self.add_copy_inst(stk_loop_var);
                    self.add_copy_inst(stk_end);
                    dprintln!("before cmp: {:?}", self.target_stack);
                    self.add_binop_inst(OpCode::Lt);
                    let jf_inst = self.add_jf_inst();
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

    fn disasm(&self, writer: &mut impl Write) -> std::io::Result<()> {
        use OpCode::*;
        writeln!(writer, "Literals [{}]", self.literals.len())?;
        for (i, con) in self.literals.iter().enumerate() {
            writeln!(writer, " [{i}] {}", *con)?;
        }

        writeln!(writer, "Instructions [{}]", self.instructions.len())?;
        for (i, inst) in self.instructions.iter().enumerate() {
            match inst.op {
                LoadLiteral => writeln!(
                    writer,
                    " [{i}] {:?} {} ({:?})",
                    inst.op, inst.arg0, self.literals[inst.arg0 as usize]
                )?,
                Copy | Call | Jmp | Jf | Pop | Store => {
                    writeln!(writer, " [{i}] {:?} {}", inst.op, inst.arg0)?
                }
                _ => writeln!(writer, " [{i}] {:?}", inst.op)?,
            }
        }
        Ok(())
    }
}

fn write_program(
    source: &str,
    writer: &mut impl Write,
    out_file: &str,
    disasm: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut compiler = Compiler::new();
    let stmts = statements_finish(source)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    dprintln!("AST: {stmts:?}");

    compiler.compile_stmts_or_zero(&stmts)?;

    if disasm {
        compiler.disasm(&mut std::io::stdout())?;
    }

    compiler.write_literals(writer).unwrap();
    compiler.write_insts(writer).unwrap();
    println!(
        "Writeen {} literals and {} instructions to {out_file:?}",
        compiler.literals.len(),
        compiler.instructions.len()
    );
    Ok(())
}

fn print_fn(args: &[Value]) -> Value {
    for arg in args {
        print!("{:?}", arg);
    }
    println!();
    Value::F64(0.)
}

struct ByteCode {
    literals: Vec<Value>,
    instructions: Vec<Instruction>,
}

impl ByteCode {
    fn new() -> Self {
        Self {
            literals: vec![],
            instructions: vec![],
        }
    }

    fn read_literals(&mut self, reader: &mut impl Read) -> std::io::Result<()> {
        let num_literals = deserialize_size(reader)?;
        for _ in 0..num_literals {
            self.literals.push(Value::deserialize(reader)?);
        }
        Ok(())
    }

    fn read_instructions(&mut self, reader: &mut impl Read) -> std::io::Result<()> {
        let num_instructions = deserialize_size(reader)?;
        for _ in 0..num_instructions {
            let inst = Instruction::deserialize(reader)?;
            self.instructions.push(inst);
        }
        Ok(())
    }

    fn interpret(&self) -> Option<Value> {
        let mut stack = vec![];
        let mut ip = 0;

        while ip < self.instructions.len() {
            let instruction = &self.instructions[ip];
            dprintln!("interpret[{ip}]: {instruction:?} stack: {stack:?}");
            match instruction.op {
                OpCode::LoadLiteral => {
                    stack.push(self.literals[instruction.arg0 as usize].clone());
                }
                OpCode::Store => {
                    let idx = stack.len() - instruction.arg0 as usize - 1;
                    let value = stack.pop().expect("Store needs an argument");
                    stack[idx] = value;
                }
                OpCode::Copy => {
                    stack.push(stack[stack.len() - instruction.arg0 as usize - 1].clone());
                }
                OpCode::Dup => {
                    let top = stack.last().unwrap().clone();
                    stack.extend((0..instruction.arg0).map(|_| top.clone()));
                }
                OpCode::Add => self.interpret_bin_op(&mut stack, |lhs, rhs| lhs + rhs),
                OpCode::Sub => self.interpret_bin_op(&mut stack, |lhs, rhs| lhs - rhs),
                OpCode::Mul => self.interpret_bin_op(&mut stack, |lhs, rhs| lhs * rhs),
                OpCode::Div => self.interpret_bin_op(&mut stack, |lhs, rhs| lhs / rhs),
                OpCode::Call => {
                    let args = &stack[stack.len() - instruction.arg0 as usize..];
                    let fname = &stack[stack.len() - instruction.arg0 as usize - 1];
                    let Value::Str(fname) = fname else {
                        panic!("Function name shall be a string: {fname:?}");
                    };
                    let res = match fname as &str {
                        "sqrt" => unary_fn(f64::sqrt)(args),
                        "sin" => unary_fn(f64::sin)(args),
                        "cos" => unary_fn(f64::cos)(args),
                        "tan" => unary_fn(f64::tan)(args),
                        "asin" => unary_fn(f64::asin)(args),
                        "acos" => unary_fn(f64::acos)(args),
                        "atan" => unary_fn(f64::atan)(args),
                        "atan2" => binary_fn(f64::atan2)(args),
                        "pow" => binary_fn(f64::powf)(args),
                        "exp" => unary_fn(f64::exp)(args),
                        "log" => binary_fn(f64::log)(args),
                        "log10" => unary_fn(f64::log10)(args),
                        "print" => print_fn(args),
                        _ => panic!("Unknown function name {fname:?}"),
                    };
                    stack.resize(stack.len() - instruction.arg0 as usize - 1, Value::F64(0.));
                    stack.push(res);
                }
                OpCode::Jmp => {
                    ip = instruction.arg0 as usize;
                    continue;
                }
                OpCode::Jf => {
                    let cond = stack.pop().expect("Jf needs an argument");
                    if cond.coerce_f64() == 0. {
                        ip = instruction.arg0 as usize;
                        continue;
                    }
                }
                OpCode::Lt => {
                    self.interpret_bin_op(&mut stack, |lhs, rhs| (lhs < rhs) as i32 as f64)
                }
                OpCode::Pop => {
                    stack.resize(stack.len() - instruction.arg0 as usize, Value::default());
                }
            }
            ip += 1
        }

        stack.pop()
    }

    fn interpret_bin_op(&self, stack: &mut Vec<Value>, op: impl FnOnce(f64, f64) -> f64) {
        let rhs = stack.pop().expect("Stack underflow").coerce_f64();
        let lhs = stack.pop().expect("Stack underflow").coerce_f64();
        stack.push(Value::F64(op(lhs, rhs)));
    }
}

fn unary_fn(f: fn(f64) -> f64) -> impl Fn(&[Value]) -> Value {
    move |args| {
        let arg = args.first().expect("functino missing argument");
        let ret = f(arg.coerce_f64());
        Value::F64(ret)
    }
}

fn binary_fn(f: fn(f64, f64) -> f64) -> impl Fn(&[Value]) -> Value {
    move |args| {
        let mut args = args.iter();
        let lhs = args
            .next()
            .expect("function missing the first argument")
            .coerce_f64();
        let rhs = args
            .next()
            .expect("function missing the second argument")
            .coerce_f64();
        Value::F64(f(lhs, rhs))
    }
}

fn compile(writer: &mut impl Write, args: &Args, out_file: &str) -> Result<(), Box<dyn Error>> {
    let src = args.source.as_ref().ok_or_else(|| {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Please specify source file to compile after -c".to_string(),
        ))
    })?;
    let source = std::fs::read_to_string(src)?;
    write_program(&source, writer, out_file, args.disasm)
}

fn read_program(reader: &mut impl Read) -> std::io::Result<ByteCode> {
    let mut bytecode = ByteCode::new();
    bytecode.read_literals(reader)?;
    bytecode.read_instructions(reader)?;
    Ok(bytecode)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(args) = parse_args(true) else {
        return Ok(());
    };

    match args.run_mode {
        RunMode::Compile => {
            let writer = std::fs::File::create(&args.output)?;
            let mut writer = BufWriter::new(writer);
            compile(&mut writer, &args, &args.output)?;
        }
        RunMode::Run(code_file) => {
            let reader = std::fs::File::open(&code_file)?;
            let mut reader = BufReader::new(reader);
            let bytecode = read_program(&mut reader)?;
            bytecode.interpret();
        }
        RunMode::CompileAndRun => {
            let mut buf = vec![];
            compile(&mut std::io::Cursor::new(&mut buf), &args, "<Memory>")?;
            let bytecode = read_program(&mut std::io::Cursor::new(&mut buf))?;
            bytecode.interpret();
        }
        _ => println!("Please specify -c, -r or -R as an argument"),
    }
    Ok(())
}

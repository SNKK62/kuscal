use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1, none_of},
    combinator::{cut, map_res, opt, recognize},
    error::ParseError,
    multi::{fold_many0, many0, separated_list0},
    number::complete::recognize_float,
    sequence::{delimited, pair, preceded, terminated},
    Finish, IResult, InputTake, Offset, Parser,
};
use nom_locate::LocatedSpan;
use std::{collections::HashMap, error::Error};

use crate::value::Value;

pub type Functions<'src> = HashMap<String, FnDecl<'src>>;

fn unary_fn<'a>(f: fn(f64) -> f64) -> FnDecl<'a> {
    FnDecl::Native(NativeFn {
        args: vec![("arg", TypeDecl::F64)],
        ret_type: TypeDecl::F64,
        code: Box::new(move |args| {
            Value::F64(f(args
                .iter()
                .next()
                .expect("function missing argument")
                .coerce_f64()))
        }),
    })
}

fn binary_fn<'a>(f: fn(f64, f64) -> f64) -> FnDecl<'a> {
    FnDecl::Native(NativeFn {
        args: vec![("lhs", TypeDecl::F64), ("rhs", TypeDecl::F64)],
        ret_type: TypeDecl::F64,
        code: Box::new(move |args| {
            let mut args = args.iter();
            let lhs = args.next().expect("function missing argument").coerce_f64();
            let rhs = args.next().expect("function missing argument").coerce_f64();
            Value::F64(f(lhs, rhs))
        }),
    })
}

fn print_fn(args: &[Value]) -> Value {
    for arg in args {
        print!("{} ", arg);
    }
    println!();
    Value::F64(0.)
}

fn dbg_fn(values: &[Value]) -> Value {
    println!("dbg: {:?}", values[0]);
    Value::I64(0)
}

fn puts_fn(args: &[Value]) -> Value {
    for arg in args {
        print!("{}", arg);
    }
    Value::F64(0.)
}

pub fn standard_functions<'src>() -> Functions<'src> {
    let mut funcs = Functions::new();
    funcs.insert("sqrt".to_string(), unary_fn(f64::sqrt));
    funcs.insert("sin".to_string(), unary_fn(f64::sin));
    funcs.insert("cos".to_string(), unary_fn(f64::cos));
    funcs.insert("tan".to_string(), unary_fn(f64::tan));
    funcs.insert("asin".to_string(), unary_fn(f64::asin));
    funcs.insert("acos".to_string(), unary_fn(f64::acos));
    funcs.insert("atan".to_string(), unary_fn(f64::atan));
    funcs.insert("atan2".to_string(), binary_fn(f64::atan2));
    funcs.insert("pow".to_string(), binary_fn(f64::powf));
    funcs.insert("exp".to_string(), unary_fn(f64::exp));
    funcs.insert("log".to_string(), binary_fn(f64::log));
    funcs.insert("log10".to_string(), unary_fn(f64::log10));
    funcs.insert(
        "print".to_string(),
        FnDecl::Native(NativeFn {
            args: vec![("arg", TypeDecl::Any)],
            ret_type: TypeDecl::Any,
            code: Box::new(print_fn),
        }),
    );
    funcs.insert(
        "dbg".to_string(),
        FnDecl::Native(NativeFn {
            args: vec![("arg", TypeDecl::Any)],
            ret_type: TypeDecl::Any,
            code: Box::new(dbg_fn),
        }),
    );
    funcs.insert(
        "puts".to_string(),
        FnDecl::Native(NativeFn {
            args: vec![("arg", TypeDecl::Any)],
            ret_type: TypeDecl::Any,
            code: Box::new(puts_fn),
        }),
    );
    funcs.insert(
        "i64".to_string(),
        FnDecl::Native(NativeFn {
            args: vec![("arg", TypeDecl::Any)],
            ret_type: TypeDecl::I64,
            code: Box::new(move |args| {
                Value::I64(
                    args.first()
                        .expect("function missing argument")
                        .coerce_i64(),
                )
            }),
        }),
    );
    funcs.insert(
        "f64".to_string(),
        FnDecl::Native(NativeFn {
            args: vec![("arg", TypeDecl::Any)],
            ret_type: TypeDecl::F64,
            code: Box::new(move |args| {
                Value::F64(
                    args.first()
                        .expect("function missing argument")
                        .coerce_f64(),
                )
            }),
        }),
    );
    funcs.insert(
        "str".to_string(),
        FnDecl::Native(NativeFn {
            args: vec![("arg", TypeDecl::Any)],
            ret_type: TypeDecl::Str,
            code: Box::new(move |args| {
                Value::Str(
                    args.first()
                        .expect("function missing argument")
                        .coerce_str(),
                )
            }),
        }),
    );
    funcs
}

pub type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq, Clone)]
pub enum TypeDecl {
    Any,
    F64,
    I64,
    Str,
    Array, // INFO: only f64 array is supported
    Coro,
}

fn tc_coerce_type<'src>(
    value: &TypeDecl,
    target: &TypeDecl,
    span: Span<'src>,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    use TypeDecl::*;
    Ok(match (value, target) {
        (_, Any) => value.clone(),
        (Any, _) => value.clone(),
        (F64 | I64, F64) => F64,
        (F64, I64) => F64,
        (I64, I64) => I64,
        (Str, Str) => Str,
        (Coro, Coro) => Coro,
        (Array, Array) => Array,
        _ => {
            return Err(TypeCheckError::new(
                format!("{:?} cannot be assigned to {:?}", value, target),
                span,
            ))
        }
    })
}

pub struct TypeCheckContext<'src, 'ctx> {
    /// Variables table for type checking.
    vars: HashMap<&'src str, TypeDecl>,
    /// Function names are owned strings because it can be either from source or native.
    funcs: Functions<'src>,
    super_context: Option<&'ctx TypeCheckContext<'src, 'ctx>>,
}

impl<'src, 'ctx> TypeCheckContext<'src, 'ctx> {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            funcs: standard_functions(),
            super_context: None,
        }
    }

    pub fn get_var(&self, name: &str) -> Option<TypeDecl> {
        match self.vars.get(name) {
            Some(val) => Some(val.clone()),
            None => match self.super_context {
                Some(super_ctx) => super_ctx.get_var(name),
                None => None,
            },
        }
    }

    pub fn get_fn(&self, name: &str) -> Option<&FnDecl<'src>> {
        if let Some(val) = self.funcs.get(name) {
            Some(val)
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_fn(name)
        } else {
            None
        }
    }

    fn push_stack(super_ctx: &'ctx Self) -> Self {
        Self {
            vars: HashMap::new(),
            funcs: HashMap::new(),
            super_context: Some(super_ctx),
        }
    }
}

#[derive(Debug)]
pub struct TypeCheckError<'src> {
    msg: String,
    pub span: Span<'src>,
}

impl<'src> std::fmt::Display for TypeCheckError<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\nlocation: {}:{}: {}",
            self.msg,
            self.span.location_line(),
            self.span.get_utf8_column(),
            self.span.fragment()
        )
    }
}

impl<'src> Error for TypeCheckError<'src> {}

impl<'src> TypeCheckError<'src> {
    fn new(msg: String, span: Span<'src>) -> Self {
        Self { msg, span }
    }
}

fn tc_binary_op<'src>(
    lhs: &Expression<'src>,
    rhs: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
    op: &str,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    let lhst = tc_expr(lhs, ctx)?;
    let rhst = tc_expr(rhs, ctx)?;
    binary_op_type(&lhst, &rhst).map_err(|_| {
        TypeCheckError::new(
            format!(
                "Operation {op} between incompatible type: {:?} and {:?}",
                lhst, rhst
            ),
            lhs.span,
        )
    })
}

fn binary_op_type(lhs: &TypeDecl, rhs: &TypeDecl) -> Result<TypeDecl, ()> {
    use TypeDecl::*;
    Ok(match (lhs, rhs) {
        (Any, _) => Any,
        (_, Any) => Any,
        (F64 | I64, I64 | F64) => F64,
        (Str, Str) => Str,
        _ => return Err(()),
    })
}

fn tc_binary_cmp<'src>(
    lhs: &Expression<'src>,
    rhs: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
    op: &str,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    use TypeDecl::*;
    let lhst = tc_expr(lhs, ctx)?;
    let rhst = tc_expr(rhs, ctx)?;
    Ok(match (&lhst, &rhst) {
        (Any, _) => I64,
        (_, Any) => I64,
        (I64, I64) => I64,
        (I64 | F64, I64 | F64) => I64, // TODO: should be cmp between same types
        (Str, Str) => I64,
        _ => {
            return Err(TypeCheckError::new(
                format!(
                    "Operation {op} bwetween incompatible type: {:?} and {:?}",
                    lhst, rhst
                ),
                lhs.span,
            ))
        }
    })
}

fn tc_expr<'src>(
    e: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    use ExprEnum::*;
    Ok(match &e.expr {
        NumLiteral(_val) => TypeDecl::F64,
        StrLiteral(_val) => TypeDecl::Str,
        ArrayLiteral(_val) => TypeDecl::Array,
        ArrayIndexAccess(name, index, ..) => {
            let var = ctx.get_var(name).ok_or_else(|| {
                TypeCheckError::new(format!("Variable \"{}\" not found", name), e.span)
            })?;
            // TODO: index should be i64
            if TypeDecl::Array == var
                && (tc_expr(index, ctx)? == TypeDecl::F64 || tc_expr(index, ctx)? == TypeDecl::I64)
            {
                TypeDecl::F64
            } else {
                return Err(TypeCheckError::new(
                    format!("Variable \"{}\" is not an array", name),
                    e.span,
                ));
            }
        }
        Ident(name) => ctx.get_var(name).ok_or_else(|| {
            TypeCheckError::new(format!("Variable \"{}\" not found", name), e.span)
        })?,
        FnInvoke(name, args) => {
            let args_ty = args
                .iter()
                .map(|v| Ok((tc_expr(v, ctx)?, v.span)))
                .collect::<Result<Vec<_>, _>>()?;
            let func = ctx.get_fn(name).ok_or_else(|| {
                TypeCheckError::new(format!("function {} is not defined", name), *name)
            })?;
            let args_decl = func.args();
            for ((arg_ty, arg_span), decl) in args_ty.iter().zip(args_decl.iter()) {
                tc_coerce_type(arg_ty, &decl.1, *arg_span)?;
            }
            func.ret_type()
        }
        Add(lhs, rhs) => tc_binary_op(lhs, rhs, ctx, "Add")?,
        Sub(lhs, rhs) => tc_binary_op(lhs, rhs, ctx, "Sub")?,
        Mul(lhs, rhs) => tc_binary_op(lhs, rhs, ctx, "Mul")?,
        Div(lhs, rhs) => tc_binary_op(lhs, rhs, ctx, "Div")?,
        Gt(lhs, rhs) => tc_binary_cmp(lhs, rhs, ctx, "GT")?,
        Lt(lhs, rhs) => tc_binary_cmp(lhs, rhs, ctx, "LT")?,
        Eq(lhs, rhs) => tc_binary_cmp(lhs, rhs, ctx, "Eq")?,
        Neq(lhs, rhs) => tc_binary_cmp(lhs, rhs, ctx, "Neq")?,
        Not(ex) => {
            let ty = tc_expr(ex, ctx)?;
            if ty == TypeDecl::I64 {
                TypeDecl::I64
            } else if ty == TypeDecl::F64 {
                TypeDecl::F64
            } else {
                return Err(TypeCheckError::new(
                    format!("Operation Not between incompatible type: {:?}", ty),
                    e.span,
                ));
            }
        }
        If(cond, true_branch, false_branch) => {
            tc_coerce_type(&tc_expr(cond, ctx)?, &TypeDecl::I64, cond.span)?;
            let true_type = type_check(true_branch, ctx)?;
            if let Some(false_branch) = false_branch {
                let false_type = type_check(false_branch, ctx)?;
                binary_op_type(&true_type, &false_type).map_err(|_| {
                    let true_span = true_branch.span();
                    let false_span = false_branch.span();
                    TypeCheckError::new(
                        format!("Conditional expression doesn't have the compatible types in true and flse banch: {:?} and {:?}", true_type, false_type),
                        calc_offset(true_span, false_span),
                    )
                })?
            } else {
                true_type
            }
        }
        Await(ex) => {
            let _res = tc_expr(ex, ctx)?;
            TypeDecl::Any
        }
    })
}

pub fn type_check<'src>(
    stmts: &Vec<Statement<'src>>,
    ctx: &mut TypeCheckContext<'src, '_>,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    let mut res = TypeDecl::Any;
    for stmt in stmts {
        match stmt {
            Statement::VarDef { name, td, ex, .. } => {
                let init_type = tc_expr(ex, ctx)?;
                let init_type = tc_coerce_type(&init_type, td, ex.span)?;
                ctx.vars.insert(**name, init_type);
            }
            Statement::VarAssign { name, ex, .. } => {
                let init_type = tc_expr(ex, ctx)?;
                let target = ctx.vars.get(**name).expect("Variable not found");
                tc_coerce_type(&init_type, target, ex.span)?;
            }
            Statement::ArrayIndexAssign { name, ex, .. } => {
                let init_type = tc_expr(ex, ctx)?;
                // TODO: now only f64 array is supported
                let _ = ctx.vars.get(**name).expect("Variable not found");
                tc_coerce_type(&init_type, &TypeDecl::F64, ex.span)?;
            }
            Statement::FnDef {
                name,
                args,
                ret_type,
                stmts,
                cofn,
            } => {
                ctx.funcs.insert(
                    name.to_string(),
                    FnDecl::User(UserFn {
                        args: args.clone(),
                        ret_type: ret_type.clone(),
                        cofn: *cofn,
                    }),
                );
                let mut subctx = TypeCheckContext::push_stack(ctx);
                for (arg, ty) in args.iter() {
                    subctx.vars.insert(arg, ty.clone());
                }
                let last_stmt = type_check(stmts, &mut subctx)?;
                tc_coerce_type(&last_stmt, ret_type, stmts.span())?;
            }
            Statement::Expression(e) => {
                res = tc_expr(e, ctx)?;
            }
            Statement::For {
                loop_var,
                start,
                end,
                stmts,
                ..
            } => {
                tc_coerce_type(&tc_expr(start, ctx)?, &TypeDecl::I64, start.span)?;
                tc_coerce_type(&tc_expr(end, ctx)?, &TypeDecl::I64, end.span)?;
                ctx.vars.insert(loop_var, TypeDecl::I64);
                res = type_check(stmts, ctx)?;
            }
            Statement::While { cond, stmts, .. } => {
                tc_coerce_type(&tc_expr(cond, ctx)?, &TypeDecl::I64, cond.span)?;
                res = type_check(stmts, ctx)?;
            }
            Statement::Return(e) => {
                return tc_expr(e, ctx);
            }
            Statement::Break => {
                // TODO
            }
            Statement::Continue => (),
            Statement::Yield(e) => {
                tc_expr(e, ctx)?;
                // TODO
            }
        }
    }
    Ok(res)
}

pub enum FnDecl<'src> {
    User(UserFn<'src>),
    Native(NativeFn<'src>),
}

impl<'src> FnDecl<'src> {
    fn args(&self) -> Vec<(&'src str, TypeDecl)> {
        match self {
            Self::User(user) => user
                .args
                .iter()
                .map(|(name, ty)| (*name.fragment(), ty.clone()))
                .collect(),
            Self::Native(native) => native.args.clone(),
        }
    }

    fn ret_type(&self) -> TypeDecl {
        match self {
            Self::User(user) => {
                if user.cofn {
                    TypeDecl::Coro
                } else {
                    user.ret_type.clone()
                }
            }
            Self::Native(native) => native.ret_type.clone(),
        }
    }
}

pub struct UserFn<'src> {
    args: Vec<(Span<'src>, TypeDecl)>,
    ret_type: TypeDecl,
    cofn: bool,
}

type NativeFnCode = dyn Fn(&[Value]) -> Value;
pub struct NativeFn<'src> {
    args: Vec<(&'src str, TypeDecl)>,
    ret_type: TypeDecl,
    pub code: Box<NativeFnCode>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum ExprEnum<'src> {
    Ident(Span<'src>),
    NumLiteral(f64),
    StrLiteral(String),
    ArrayLiteral(Vec<f64>), // INFO: only f64 array is supported
    ArrayIndexAccess(Span<'src>, Box<Expression<'src>>),
    FnInvoke(Span<'src>, Vec<Expression<'src>>),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
    Sub(Box<Expression<'src>>, Box<Expression<'src>>),
    Mul(Box<Expression<'src>>, Box<Expression<'src>>),
    Div(Box<Expression<'src>>, Box<Expression<'src>>),
    Gt(Box<Expression<'src>>, Box<Expression<'src>>),
    Lt(Box<Expression<'src>>, Box<Expression<'src>>),
    Eq(Box<Expression<'src>>, Box<Expression<'src>>),
    Neq(Box<Expression<'src>>, Box<Expression<'src>>),
    Not(Box<Expression<'src>>),
    If(
        Box<Expression<'src>>,
        Box<Statements<'src>>,
        Option<Box<Statements<'src>>>,
    ),
    Await(Box<Expression<'src>>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Expression<'a> {
    pub(crate) expr: ExprEnum<'a>,
    pub(crate) span: Span<'a>,
}

impl<'a> Expression<'a> {
    fn new(expr: ExprEnum<'a>, span: Span<'a>) -> Self {
        Self { expr, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement<'src> {
    Expression(Expression<'src>),
    VarDef {
        span: Span<'src>,
        name: Span<'src>,
        td: TypeDecl,
        ex: Expression<'src>,
    },
    VarAssign {
        span: Span<'src>,
        name: Span<'src>,
        ex: Expression<'src>,
    },
    ArrayIndexAssign {
        span: Span<'src>,
        name: Span<'src>,
        index: Expression<'src>,
        ex: Expression<'src>,
    },
    For {
        span: Span<'src>,
        loop_var: Span<'src>,
        start: Expression<'src>,
        end: Expression<'src>,
        stmts: Statements<'src>,
    },
    While {
        span: Span<'src>,
        cond: Expression<'src>,
        stmts: Statements<'src>,
    },
    Break,
    Continue,
    FnDef {
        name: Span<'src>,
        args: Vec<(Span<'src>, TypeDecl)>,
        ret_type: TypeDecl,
        stmts: Statements<'src>,
        cofn: bool,
    },
    Return(Expression<'src>),
    Yield(Expression<'src>),
}

impl<'src> Statement<'src> {
    fn span(&self) -> Option<Span<'src>> {
        use Statement::*;
        Some(match self {
            Expression(ex) => ex.span,
            VarDef { span, .. } => *span,
            VarAssign { span, .. } => *span,
            ArrayIndexAssign { span, .. } => *span,
            For { span, .. } => *span,
            While { span, .. } => *span,
            FnDef { name, stmts, .. } => calc_offset(*name, stmts.span()),
            Return(ex) => ex.span,
            Break | Continue => return None,
            Yield(ex) => ex.span,
        })
    }
}

trait GetSpan<'a> {
    fn span(&self) -> Span<'a>;
}

pub type Statements<'src> = Vec<Statement<'src>>;

impl<'a> GetSpan<'a> for Statements<'a> {
    fn span(&self) -> Span<'a> {
        self.iter().find_map(|stmt| stmt.span()).unwrap()
    }
}

fn space_delimited<'src, O, E>(
    f: impl Parser<Span<'src>, O, E>,
) -> impl FnMut(Span<'src>) -> IResult<Span<'src>, O, E>
where
    E: ParseError<Span<'src>>,
{
    delimited(multispace0, f, multispace0)
}

fn calc_offset<'a>(i: Span<'a>, r: Span<'a>) -> Span<'a> {
    i.take(i.offset(&r))
}

fn factor(i: Span) -> IResult<Span, Expression> {
    alt((
        str_literal,
        num_literal,
        func_call,
        array_index_access,
        ident,
        not_factor,
        parens,
    ))(i)
}

fn func_call(i: Span) -> IResult<Span, Expression> {
    let (r, ident) = space_delimited(identifier)(i)?;
    let (r, args) = space_delimited(delimited(
        tag("("),
        many0(delimited(multispace0, expr, space_delimited(opt(tag(","))))),
        tag(")"),
    ))(r)?;
    Ok((
        r,
        Expression {
            expr: ExprEnum::FnInvoke(ident, args),
            span: i,
        },
    ))
}

fn term(input: Span) -> IResult<Span, Expression> {
    let (r, init) = factor(input)?;

    let res = fold_many0(
        pair(space_delimited(alt((char('*'), char('/')))), factor),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| {
            let span = calc_offset(input, acc.span);
            match op {
                '*' => Expression::new(ExprEnum::Mul(Box::new(acc), Box::new(val)), span),
                '/' => Expression::new(ExprEnum::Div(Box::new(acc), Box::new(val)), span),
                _ => panic!("Multiplicative expression should have '*' or '/' operator"),
            }
        },
    )(r);
    res
}

fn ident(input: Span) -> IResult<Span, Expression> {
    let (r, res) = space_delimited(identifier)(input)?;
    Ok((
        r,
        Expression {
            expr: ExprEnum::Ident(res),
            span: input,
        },
    ))
}

fn identifier(input: Span) -> IResult<Span, Span> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn str_literal(i: Span) -> IResult<Span, Expression> {
    let (r0, _) = preceded(multispace0, char('\"'))(i)?;
    let (r, val) = many0(none_of("\""))(r0)?;
    let (r, _) = terminated(char('"'), multispace0)(r)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::StrLiteral(
                val.iter()
                    .collect::<String>()
                    .replace("\\\\", "\\")
                    .replace("\\n", "\n"),
            ),
            i,
        ),
    ))
}

fn num_literal(input: Span) -> IResult<Span, Expression> {
    let (r, v) = space_delimited(recognize_float)(input)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::NumLiteral(v.parse().map_err(|_| {
                nom::Err::Error(nom::error::Error {
                    input,
                    code: nom::error::ErrorKind::Digit,
                })
            })?),
            v,
        ),
    ))
}

fn array_literal(input: Span) -> IResult<Span, Expression> {
    let (r, _) = preceded(multispace0, char('['))(input)?;
    let (r, (v, span)) = cut(|i| {
        let (i, v) =
            separated_list0(space_delimited(char(',')), space_delimited(recognize_float))(i)?;
        let (i, _) = space_delimited(char(']'))(i)?;
        Ok((i, (v.iter().map(|s| s.parse().unwrap()).collect(), v)))
    })(r)?;
    Ok((
        r,
        Expression::new(ExprEnum::ArrayLiteral(v), span.last().cloned().unwrap()), // TODO: fix span
    ))
}

fn array_index_access(i: Span) -> IResult<Span, Expression> {
    let (r, name) = space_delimited(identifier)(i)?;
    let (r, _) = space_delimited(char('['))(r)?;
    // TODO: to be nom::combinator::cut
    let (r, index) = space_delimited(expr)(r)?;
    let (r, _) = space_delimited(char(']'))(r)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::ArrayIndexAccess(name, Box::new(index)), // TODO: fix index
            i,
        ),
    ))
}

fn parens(i: Span) -> IResult<Span, Expression> {
    space_delimited(delimited(tag("("), expr, tag(")")))(i)
}

fn not_factor(i: Span) -> IResult<Span, Expression> {
    let (i, _) = space_delimited(tag("!"))(i)?;
    let (i, cond) = cut(factor)(i)?;
    Ok((i, Expression::new(ExprEnum::Not(Box::new(cond)), i)))
}

fn num_expr(i: Span) -> IResult<Span, Expression> {
    let (r, init) = term(i)?;

    let res = fold_many0(
        pair(space_delimited(alt((char('+'), char('-')))), term),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| {
            let span = calc_offset(i, acc.span);
            match op {
                '+' => Expression::new(ExprEnum::Add(Box::new(acc), Box::new(val)), span),
                '-' => Expression::new(ExprEnum::Sub(Box::new(acc), Box::new(val)), span),
                _ => panic!("Additive expression should have '+' or '-' operator"),
            }
        },
    )(r);
    res
}

fn cond_expr(i0: Span) -> IResult<Span, Expression> {
    let (i, first) = num_expr(i0)?;
    let (i, cond) = space_delimited(alt((tag("<"), tag(">"), tag("=="), tag("!="))))(i)?;
    let (i, second) = num_expr(i)?;
    let span = calc_offset(i0, i);
    Ok((
        i,
        match *cond.fragment() {
            "<" => Expression::new(ExprEnum::Lt(Box::new(first), Box::new(second)), span),
            ">" => Expression::new(ExprEnum::Gt(Box::new(first), Box::new(second)), span),
            "==" => Expression::new(ExprEnum::Eq(Box::new(first), Box::new(second)), span),
            "!=" => Expression::new(ExprEnum::Neq(Box::new(first), Box::new(second)), span),
            _ => unreachable!(),
        },
    ))
}

fn open_brace(i: Span) -> IResult<Span, ()> {
    let (i, _) = space_delimited(char('{'))(i)?;
    Ok((i, ()))
}

fn close_brace(i: Span) -> IResult<Span, ()> {
    let (i, _) = space_delimited(char('}'))(i)?;
    Ok((i, ()))
}

fn if_expr(i0: Span) -> IResult<Span, Expression> {
    let (i, _) = space_delimited(tag("if"))(i0)?;
    let (i, cond) = expr(i)?;
    let (i, t_case) = delimited(open_brace, statements, close_brace)(i)?;
    let (i, f_case) = opt(preceded(
        space_delimited(tag("else")),
        alt((
            delimited(open_brace, statements, close_brace),
            map_res(
                if_expr,
                |v| -> Result<Vec<Statement>, nom::error::Error<&str>> {
                    Ok(vec![Statement::Expression(v)])
                },
            ),
        )),
    ))(i)?;

    Ok((
        i,
        Expression::new(
            ExprEnum::If(Box::new(cond), Box::new(t_case), f_case.map(Box::new)),
            calc_offset(i0, i),
        ),
    ))
}

fn await_expr(i: Span) -> IResult<Span, Expression> {
    let i0 = i;
    let (i, _) = space_delimited(tag("await"))(i)?;
    let (i, ex) = cut(space_delimited(expr))(i)?;
    Ok((
        i,
        Expression::new(ExprEnum::Await(Box::new(ex)), calc_offset(i0, i)),
    ))
}

pub fn expr(i: Span) -> IResult<Span, Expression> {
    alt((await_expr, if_expr, cond_expr, num_expr, array_literal))(i)
}

fn var_def(i: Span) -> IResult<Span, Statement> {
    let span = i;
    let (i, _) = delimited(multispace0, tag("var"), multispace1)(i)?;
    let (i, (name, td, ex)) = cut(|i| {
        let (i, name) = space_delimited(identifier)(i)?;
        let (i, _) = space_delimited(char(':'))(i)?;
        let (i, td) = type_decl(i)?;
        let (i, _) = space_delimited(char('='))(i)?;
        let (i, ex) = space_delimited(expr)(i)?;
        let (i, _) = space_delimited(char(';'))(i)?;
        Ok((i, (name, td, ex)))
    })(i)?;
    Ok((
        i,
        Statement::VarDef {
            span: calc_offset(span, i),
            name,
            td,
            ex,
        },
    ))
}

fn var_assign(i: Span) -> IResult<Span, Statement> {
    let span = i;
    let (i, name) = space_delimited(identifier)(i)?;
    let (i, _) = space_delimited(char('='))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    let (i, _) = space_delimited(char(';'))(i)?;
    Ok((
        i,
        Statement::VarAssign {
            span: calc_offset(span, i),
            name,
            ex,
        },
    ))
}

fn array_index_assign(i: Span) -> IResult<Span, Statement> {
    let span = i;
    let (i, name) = space_delimited(identifier)(i)?;
    let (i, _) = space_delimited(char('['))(i)?;
    // TODO: to be nom::combinator::cut
    let (i, index) = space_delimited(expr)(i)?;
    let (i, _) = space_delimited(char(']'))(i)?;
    let (i, _) = space_delimited(char('='))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    let (i, _) = space_delimited(char(';'))(i)?;
    Ok((
        i,
        Statement::ArrayIndexAssign {
            span: calc_offset(span, i),
            name,
            index,
            ex,
        },
    ))
}

fn expr_statement(i: Span) -> IResult<Span, Statement> {
    let (i, res) = expr(i)?;
    Ok((i, Statement::Expression(res)))
}

fn for_statement(i: Span) -> IResult<Span, Statement> {
    let i0 = i;
    let (i, _) = space_delimited(tag("for"))(i)?;
    let (i, (loop_var, start, end, stmts)) = cut(|i| {
        let (i, loop_var) = space_delimited(identifier)(i)?;
        let (i, _) = space_delimited(tag("in"))(i)?;
        let (i, start) = space_delimited(expr)(i)?;
        let (i, _) = space_delimited(tag("to"))(i)?;
        let (i, end) = space_delimited(expr)(i)?;
        let (i, stmts) = delimited(open_brace, statements, close_brace)(i)?;
        Ok((i, (loop_var, start, end, stmts)))
    })(i)?;
    Ok((
        i,
        Statement::For {
            span: calc_offset(i0, i),
            loop_var,
            start,
            end,
            stmts,
        },
    ))
}

fn while_statement(i: Span) -> IResult<Span, Statement> {
    let i0 = i;
    let (i, _) = space_delimited(tag("while"))(i)?;
    let (i, (cond, stmts)) = cut(|i| {
        let (i, cond) = space_delimited(expr)(i)?;
        let (i, stmts) = delimited(open_brace, statements, close_brace)(i)?;
        Ok((i, (cond, stmts)))
    })(i)?;
    Ok((
        i,
        Statement::While {
            span: calc_offset(i0, i),
            cond,
            stmts,
        },
    ))
}

fn type_decl(i: Span) -> IResult<Span, TypeDecl> {
    let (i, td) = space_delimited(identifier)(i)?;
    Ok((
        i,
        match *td.fragment() {
            "i64" => TypeDecl::I64,
            "f64" => TypeDecl::F64,
            "str" => TypeDecl::Str,
            "Array" => TypeDecl::Array,
            "cofn" => TypeDecl::Coro,
            _ => {
                return Err(nom::Err::Failure(nom::error::Error::new(
                    td,
                    nom::error::ErrorKind::Verify,
                )))
            }
        },
    ))
}

fn argument(i: Span) -> IResult<Span, (Span, TypeDecl)> {
    let (i, ident) = space_delimited(identifier)(i)?;
    let (i, _) = char(':')(i)?;
    let (i, td) = type_decl(i)?;
    Ok((i, (ident, td)))
}

fn fn_def_statement(i: Span) -> IResult<Span, Statement> {
    let (i, fn_kw) = space_delimited(alt((tag("cofn"), tag("fn"))))(i)?;
    let (i, (name, args, ret_type, stmts)) = cut(|i| {
        let (i, name) = space_delimited(identifier)(i)?;
        let (i, _) = space_delimited(tag("("))(i)?;
        let (i, args) = separated_list0(char(','), space_delimited(argument))(i)?;
        let (i, _) = space_delimited(tag(")"))(i)?;
        let (i, _) = space_delimited(tag("->"))(i)?;
        let (i, ret_type) = type_decl(i)?;
        let (i, stmts) = delimited(open_brace, statements, close_brace)(i)?;

        Ok((i, (name, args, ret_type, stmts)))
    })(i)?;
    Ok((
        i,
        Statement::FnDef {
            name,
            args,
            ret_type,
            stmts,
            cofn: *fn_kw == "cofn",
        },
    ))
}

fn return_statement(i: Span) -> IResult<Span, Statement> {
    let (i, _) = space_delimited(tag("return"))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    Ok((i, Statement::Return(ex)))
}

fn break_statement(i: Span) -> IResult<Span, Statement> {
    let (i, _) = space_delimited(tag("break"))(i)?;
    Ok((i, Statement::Break))
}

fn continue_statement(i: Span) -> IResult<Span, Statement> {
    let (i, _) = space_delimited(tag("continue"))(i)?;
    Ok((i, Statement::Continue))
}

fn yield_statement(i: Span) -> IResult<Span, Statement> {
    let (i, _) = space_delimited(tag("yield"))(i)?;
    let (i, ex) = cut(space_delimited(expr))(i)?;
    Ok((i, Statement::Yield(ex)))
}

fn general_statement<'a>(last: bool) -> impl Fn(Span<'a>) -> IResult<Span<'a>, Statement> {
    let terminator = move |i| -> IResult<Span, ()> {
        let mut semicolon = pair(tag(";"), multispace0);
        if last {
            Ok((opt(semicolon)(i)?.0, ()))
        } else {
            Ok((semicolon(i)?.0, ()))
        }
    };
    move |input| {
        alt((
            var_def,
            var_assign,
            array_index_assign,
            fn_def_statement,
            for_statement,
            while_statement,
            terminated(return_statement, terminator),
            terminated(break_statement, terminator),
            terminated(continue_statement, terminator),
            terminated(yield_statement, terminator),
            terminated(expr_statement, terminator),
        ))(input)
    }
}

pub fn last_statement(i: Span) -> IResult<Span, Statement> {
    general_statement(true)(i)
}

pub fn statement(i: Span) -> IResult<Span, Statement> {
    general_statement(false)(i)
}

fn statements(i: Span) -> IResult<Span, Statements> {
    let (i, mut stmts) = many0(statement)(i)?;
    let (i, last) = opt(last_statement)(i)?;
    let (i, _) = opt(multispace0)(i)?;
    if let Some(last) = last {
        stmts.push(last);
    }
    Ok((i, stmts))
}

pub fn statements_finish(i: Span) -> Result<Statements, nom::error::Error<Span>> {
    let (_, res) = statements(i).finish()?;
    Ok(res)
}

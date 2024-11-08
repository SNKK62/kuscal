// ref: https://github.com/msakuta/ruscal/blob/ed869ab38ba0608b75ec63040bcc06eb8a6fc5d7/src/lib.rs
// use std::{collections::HashMap, sync::atomic::AtomicBool};
use std::sync::atomic::AtomicBool;

pub enum RunMode {
    None,
    TypeCheck,
    Compile,
    Run(String),
    CompileAndRun,
}

pub struct Args {
    pub run_mode: RunMode,
    pub source: Option<String>,
    pub output: String,
    pub disasm: bool,
    pub show_ast: bool,
    pub show_debug_ast: bool,
    pub optimize: bool,
    pub debug_output: bool,
    // Because Args is passed as a shared reference, NativeFn can be requested to be generated multiple times.
    // Having a function to return one is an easy trick to allow it without breaking API.
    // pub additional_funcs: HashMap<String, Box<dyn Fn() -> NativeFn<'static>>>,
}

impl Args {
    pub fn new() -> Self {
        Self {
            run_mode: RunMode::CompileAndRun,
            source: None,
            output: "".to_string(),
            disasm: false,
            show_ast: false,
            show_debug_ast: false,
            optimize: false,
            debug_output: false,
            // additional_funcs: HashMap::new(),
        }
    }
}

impl Default for Args {
    fn default() -> Self {
        Self::new()
    }
}

pub static DEBUG: AtomicBool = AtomicBool::new(false);

pub fn parse_args(compilable: bool) -> Option<Args> {
    let mut run_mode = RunMode::None;
    let mut source = None;
    let mut output = None;
    let mut disasm = false;
    let mut show_ast = false;
    let mut show_debug_ast = false;
    let mut optimize = false;
    let mut show_help = false;
    let mut args_is_empty = true;
    let mut debug_output = false;

    let mut args = std::env::args();
    let exe = args.next();
    let mut next_arg = args.next();
    while let Some(arg) = next_arg {
        match &arg as &str {
            "-h" => show_help = true,
            "-c" => run_mode = RunMode::Compile,
            "-o" => output = args.next(),
            "-O" => optimize = true,
            "-r" => {
                let bytecode = args.next().unwrap_or_else(|| "bytecode.bin".to_string());
                run_mode = RunMode::Run(bytecode);
            }
            "-R" => run_mode = RunMode::CompileAndRun,
            "-d" => disasm = true,
            "-a" => show_ast = true,
            "-A" => show_debug_ast = true,
            "-t" => run_mode = RunMode::TypeCheck,
            "-D" => {
                DEBUG.store(true, std::sync::atomic::Ordering::Relaxed);
                debug_output = true;
            }
            _ => {
                if source.is_none() {
                    source = Some(arg);
                } else {
                    println!("More than 1 file name is specified");
                    return None;
                }
            }
        }
        args_is_empty = false;
        next_arg = args.next();
    }

    if show_help || args_is_empty {
        let compiler_options = if compilable {
            r#"    -c       Compile source file to a bytecode
    -o file  Specify output file
    -r       Run bytecode
    -R       Compile and run
    -d       Disassemble compiled code
    -O       Optimize code by constant folding/propagation
    -a       Show parsed AST
    -A       Show parsed AST in debug form"#
        } else {
            ""
        };
        println!(
            r#"Usage: {} [options] [source.txt]

Options:
{compiler_options}
    -a       Show AST
    -t       Typecheck only, no execution of the code
    -h       Display help
"#,
            exe.unwrap_or_else(|| "29-full-stmt".to_string())
        );
        return None;
    }

    Some(Args {
        run_mode,
        source,
        output: output.unwrap_or_else(|| "bytecode.bin".to_string()),
        disasm,
        show_ast,
        show_debug_ast,
        optimize,
        debug_output,
        // additional_funcs: HashMap::new(),
    })
}

#[macro_export]
macro_rules! dprintln {
    ($fmt:literal) => {
        #[cfg(not(target_arch = "wasm32"))]
        if ::ruscal::DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            println!($fmt);
        }
    };
    ($fmt:literal, $($args:expr),*) => {
        #[cfg(not(target_arch = "wasm32"))]
        if ::ruscal::DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            println!($fmt, $($args),*);
        }
    };
}

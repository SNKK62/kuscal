mod compiler;
mod parser;
mod value;
use compiler::{compile, debugger, read_program, Vm, YieldResult};
use ruscal::{parse_args, RunMode};
use std::{
    io::{BufReader, BufWriter},
    rc::Rc,
};
use value::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(args) = parse_args(true) else {
        return Ok(());
    };

    let run_coro = |mut vm: Vm| {
        if let Err(e) = vm.init_fn("main", &[]) {
            eprintln!("init_fn error: {e:?}");
        }
        loop {
            match vm.interpret() {
                Ok(YieldResult::Finished(_)) => break,
                Ok(YieldResult::Suspend(value)) => {
                    println!("Execution suspended with a yielded value {value}");
                    if value == Value::Str("break".to_string()) && debugger(&vm) {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Runtime error: {e:?}");
                    break;
                }
            }
        }
    };

    match args.run_mode {
        RunMode::TypeCheck => {
            if let Err(e) = compile(&mut std::io::sink(), &args, &args.output) {
                eprintln!("TypeCheck error: {e}");
            }
        }
        RunMode::Compile => {
            let writer = std::fs::File::create(&args.output)?;
            let mut writer = BufWriter::new(writer);
            if let Err(e) = compile(&mut writer, &args, &args.output) {
                eprintln!("Compile Error: {e}");
            }
        }
        RunMode::Run(code_file) => {
            let reader = std::fs::File::open(&code_file)?;
            let mut reader = BufReader::new(reader);
            let bytecode = Rc::new(read_program(&mut reader)?);
            run_coro(Vm::new(bytecode));
        }
        RunMode::CompileAndRun => {
            let mut buf = vec![];
            if let Err(e) = compile(&mut std::io::Cursor::new(&mut buf), &args, "<Memory>") {
                eprintln!("Compile error: {e}");
                return Ok(());
            }
            let bytecode = Rc::new(read_program(&mut std::io::Cursor::new(&mut buf))?);
            run_coro(Vm::new(bytecode));
        }
        _ => println!("Please specify -c, -r, -t or -R as an argument"),
    }
    Ok(())
}

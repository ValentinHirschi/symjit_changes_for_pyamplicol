//! `Application::save` writes the same bytes for every compilation of a program. Four parts of
//! the file used to change from one compilation to the next: the tags that keep external calls
//! apart (`rand::random()`), the order of the function table and of the real locations (both
//! `HashSet`s), and the order of the subroutines under `compress` (a `HashMap`). Each test covers
//! one of them.
mod common;

use common::base_config;
use symjit::{Complex, Composer, Defuns, Slot, Storage, Translator};

const POINT: [Complex<f64>; 4] = [
    Complex::new(3.0, 1.0),
    Complex::new(0.5, -2.0),
    Complex::new(-1.0, 0.25),
    Complex::new(2.0, 3.0),
];

/// Compiles out = f0(p0) + f1(p1) + f2(p2) + f3(p3) in complex mode, with the external function
/// `twice`; `real` marks the arguments as real. Returns the saved bytes and the value at POINT.
fn compile_and_save(funs: [&str; 4], real: bool) -> anyhow::Result<(Vec<u8>, Complex<f64>)> {
    let mut config = base_config();
    config.set_complex(true);
    let mut functions = Defuns::new();
    functions.add_sliced_func::<Complex<f64>>(
        "twice",
        Box::new(|arguments: &[Complex<f64>]| arguments[0] * 2.0),
    )?;
    config.set_defuns(functions);
    let mut translator = Translator::new(config);
    translator.set_num_params(4);
    let terms: Vec<Slot> = (0..4).map(Slot::Temp).collect();
    for (i, fun) in funs.iter().enumerate() {
        translator.append_fun(&terms[i], fun, &[Slot::Param(i)], real)?;
    }
    translator.append_add(&Slot::Out(0), &terms, 0)?;
    let application = translator.compile()?;
    let mut bytes = Vec::new();
    application.save(&mut bytes)?;
    let applet = application.seal()?;
    let mut out = [Complex::new(0.0, 0.0)];
    applet.evaluate(&POINT, &mut out);
    Ok((bytes, out[0]))
}

/// Compiles the program ten times; the saved bytes and the value must not change.
fn compile_ten_times(funs: [&str; 4], real: bool) -> anyhow::Result<Complex<f64>> {
    let (first, value) = compile_and_save(funs, real)?;
    for _ in 1..10 {
        let (bytes, again) = compile_and_save(funs, real)?;
        assert_eq!(again, value);
        assert!(
            bytes == first,
            "saved applications differ between compilations"
        );
    }
    Ok(value)
}

/// Four calls of one external function: distinct tags, the same in every compilation.
#[test]
fn external_call_tags() -> anyhow::Result<()> {
    let value = compile_ten_times(["twice"; 4], false)?;
    // 2 (p0 + p1 + p2 + p3): CSE has not merged the calls.
    assert_eq!(value, Complex::new(9.0, 4.5));
    Ok(())
}

/// Four built-in functions, so four entries in the function table (no external calls).
#[test]
fn function_table() -> anyhow::Result<()> {
    compile_ten_times(["sin", "cos", "exp", "tan"], false)?;
    Ok(())
}

/// Four real arguments, so four real locations (no external calls).
#[test]
fn real_locations() -> anyhow::Result<()> {
    compile_ten_times(["sin"; 4], true)?;
    Ok(())
}

/// `compress`: four expression shapes, three times each, so four subroutines (no external
/// calls); their order used to follow a `HashMap`.
#[test]
fn subroutines() -> anyhow::Result<()> {
    let compile_and_save = || -> anyhow::Result<Vec<u8>> {
        let mut config = base_config();
        config.set_compress(true);
        let mut translator = Translator::new(config);
        translator.set_num_params(9);
        let p = Slot::Param;
        for k in 0..3 {
            let (a, b, c) = (p(3 * k), p(3 * k + 1), p(3 * k + 2));
            let t = |i| Slot::Temp(4 * k + i);
            translator.append_mul(&t(0), &[a, b], 0)?;
            translator.append_add(&Slot::Out(4 * k), &[t(0), c], 0)?;
            translator.append_add(&t(1), &[a, b], 0)?;
            translator.append_mul(&Slot::Out(4 * k + 1), &[t(1), c], 0)?;
            translator.append_add(&Slot::Out(4 * k + 2), &[a, b, c], 0)?;
            translator.append_mul(&Slot::Out(4 * k + 3), &[a, b, c], 0)?;
        }
        let mut bytes = Vec::new();
        translator.compile()?.save(&mut bytes)?;
        Ok(bytes)
    };
    let first = compile_and_save()?;
    for _ in 1..10 {
        assert!(
            compile_and_save()? == first,
            "saved applications differ between compilations"
        );
    }
    Ok(())
}

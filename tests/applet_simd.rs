// SIMD kernels that call applets (`Defuns::add_applet`).

mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Applet, Composer, Config, Defuns, Slot, Translator};

/// g(x) = x + 1 for x > 0, else x - 1, with a kernel branch: the lanes of a SIMD row take
/// different sides of it for arguments of both signs.
fn branching_applet() -> Result<Applet> {
    let mut config = base_config();
    config.set_direct(true);
    config.set_simd(true);
    let mut ev = Translator::new(config);
    ev.set_num_params(1);
    ev.append_constant(Complex::new(0.0, 0.0))?;
    ev.append_constant(Complex::new(1.0, 0.0))?;
    ev.append_constant(Complex::new(-1.0, 0.0))?;
    ev.append_fun(&Slot::Temp(0), "gt", &[Slot::Param(0), Slot::Const(0)], true)?;
    ev.append_if_else(&Slot::Temp(0), 5)?;
    ev.append_add(&Slot::Temp(1), &[Slot::Param(0), Slot::Const(1)], 0)?;
    ev.append_goto(7)?;
    ev.append_label(5)?;
    ev.append_add(&Slot::Temp(2), &[Slot::Param(0), Slot::Const(2)], 0)?;
    ev.append_label(7)?;
    ev.append_join(&Slot::Out(0), &Slot::Temp(0), &Slot::Temp(1), &Slot::Temp(2))?;
    ev.compile()?.seal()
}

fn g(x: f64) -> f64 {
    if x > 0.0 {
        x + 1.0
    } else {
        x - 1.0
    }
}

fn caller_config(df: Defuns, simd512: bool) -> Config {
    let mut config = base_config();
    config.set_simd(true);
    config.set_option("enable_simd512", &simd512.to_string()).unwrap();
    config.set_defuns(df);
    config
}

// An applet without a SIMD kernel (e.g. sealed from `Compiler::compile`, which does not
// prepare one): the caller's SIMD kernel called `_simd_<name>_`, which did not exist; the
// caller now stays scalar.
#[test]
fn applet_without_simd_kernel() -> Result<()> {
    let mut cubed = {
        let mut ev = Translator::new(base_config());
        ev.set_num_params(1);
        ev.append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Param(0), Slot::Param(0)], 0)?;
        ev.compile()?.seal()?
    };
    cubed.compiled_simd = None;
    let mut df = Defuns::new();
    df.add_applet("cubed", cubed);

    for simd512 in [false, true] {
        let mut ev = Translator::new(caller_config(df.clone(), simd512));
        ev.set_num_params(1);
        ev.append_fun(&Slot::Out(0), "cubed", &[Slot::Param(0)], false)?;
        let app = ev.compile()?.seal()?;
        let args: Vec<f64> = (0..21).map(|i| i as f64 * 0.5).collect();
        let mut outs = vec![0.0; 21];
        app.evaluate_matrix(&args, &mut outs, 21);
        for (x, y) in args.iter().zip(&outs) {
            assert_eq!(*y, x * x * x);
        }
    }
    Ok(())
}

// A SIMD row whose lanes take different sides of a branch in a called applet: the applet's
// SIMD kernel stops (returns nonzero) and the caller ignored it, keeping whatever the lanes
// held; it now stops as well, and the row is computed by the scalar kernels.
#[test]
fn diverging_lanes_in_applet() -> Result<()> {
    let mut df = Defuns::new();
    df.add_applet("g", branching_applet()?);

    let xs: Vec<f64> = (0..37).map(|i| (i as f64 * 0.7548776662).fract() * 4.0 - 2.0).collect();
    for simd512 in [false, true] {
        let mut ev = Translator::new(caller_config(df.clone(), simd512));
        ev.set_num_params(1);
        ev.append_constant(Complex::new(2.0, 0.0))?;
        ev.append_fun(&Slot::Temp(0), "g", &[Slot::Param(0)], false)?;
        ev.append_mul(&Slot::Out(0), &[Slot::Temp(0), Slot::Const(0)], 0)?;
        let app = ev.compile()?.seal()?;
        let mut outs = vec![0.0; xs.len()];
        app.evaluate_matrix(&xs, &mut outs, xs.len());
        for (x, y) in xs.iter().zip(&outs) {
            assert_eq!(*y, 2.0 * g(*x), "x = {x}, simd512 = {simd512}");
        }
    }
    Ok(())
}

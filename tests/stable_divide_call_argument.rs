//! `stable_math` complex division whose quotient is passed straight to a function call: with the
//! indirect translator at opt level >= 1, the quotient's destination is `Left` (the call
//! argument), whose parts are the machine registers `Ret` and `Temp`. `Complexifier::stable_divide`
//! then puts the scaled numerator in `dst` and the divisor in `Gen(0)`, `Gen(1)`
//! (`if !matches!(dst, Reg::Gen(_))`). Its earlier choices failed here: `Ret` and `re(dst)` for
//! the divisor (`re(Left)` is `Ret`: c' overwritten by d'), and `re(dst)`, `im(dst)` (the
//! generators' `divide_complex` overwrites `Reg::Temp` = d' with its scratch), even for ordinary
//! values. Until 2026-10-20 this case had its own code, `stable_divide_general`.
//!
//! All four conditions are needed: `stable_math`, the indirect translator (`direct = false`; the
//! direct translator and `Composer` programs never give `Left` here), opt level >= 1, and the
//! quotient used only as a call argument. The Python equivalent is
//! `compile_func([x, y], [sp.sin(x / y)], dtype="complex128", stable_math=True)`.
//!
//! Checked on 2026-10-20 with a print in that branch (a scratch copy): this test reaches it 18
//! times (opt levels 1–3), always with `dst = Left`. Without the special case it fails at scale 1
//! (`sin(z / w)` = -1.015+0.045i, or 1128-974i, instead of -3.166+1.960i), with `fast_complex`
//! off.
mod common;
use common::base_config;

use num_complex::Complex;
use symjit::{Composer, Slot, Translator};

#[test]
fn stable_quotient_passed_to_a_call() -> anyhow::Result<()> {
    // z / w = (3 + 4i) / (1 - 2i) = -1 + 2i at every scale; unscaled, |w|^2 over- or
    // underflows at 1e+-300
    for scale in [1.0, 1e300, 1e-300] {
        let z = Complex::new(3.0, 4.0) * scale;
        let w = Complex::new(1.0, -2.0) * scale;
        let want = [
            Complex::new(-1.0, 2.0).sin(),
            (Complex::new(-1.0, 2.0) * 0.5).exp() * 2.0,
        ];
        for opt_level in 0..=3u8 {
            for fast_complex in [false, true] {
                let mut config = base_config();
                config.set_complex(true);
                config.set_stable_math(true);
                config.set_fast_complex(fast_complex);
                config.set_direct(false);
                config.set_opt_level(opt_level);
                let mut t = Translator::new(config);
                t.set_num_params(2);
                let two = t.append_constant(Complex::new(2.0, 0.0))?;
                // sin(z / w), and 2 exp(1 / (2 w / z)) (a real numerator and a quotient
                // divided again)
                t.append_div(
                    &Slot::Temp(0),
                    &Slot::Param(0),
                    &Slot::Param(1),
                    false,
                    false,
                )?;
                t.append_fun(&Slot::Out(0), "sin", &[Slot::Temp(0)], false)?;
                t.append_div(
                    &Slot::Temp(1),
                    &Slot::Param(1),
                    &Slot::Param(0),
                    false,
                    false,
                )?;
                t.append_mul(&Slot::Temp(2), &[Slot::Const(two), Slot::Temp(1)], 0)?;
                t.append_div(
                    &Slot::Temp(3),
                    &Slot::Const(two),
                    &Slot::Temp(2),
                    false,
                    false,
                )?;
                t.append_div(
                    &Slot::Temp(4),
                    &Slot::Temp(3),
                    &Slot::Const(two),
                    false,
                    false,
                )?;
                t.append_fun(&Slot::Temp(5), "exp", &[Slot::Temp(4)], false)?;
                t.append_mul(&Slot::Out(1), &[Slot::Temp(5), Slot::Const(two)], 0)?;
                let f = t.compile()?.seal()?;
                let mut outs = [Complex::default(); 2];
                f.evaluate(&[z, w], &mut outs);
                for k in 0..2 {
                    let err = (outs[k] - want[k]).norm() / want[k].norm();
                    assert!(
                        err < 1e-15,
                        "scale {:e}, opt level {}, fast_complex {}, output {}: {} (want {})",
                        scale,
                        opt_level,
                        fast_complex,
                        k,
                        outs[k],
                        want[k]
                    );
                }
            }
        }
    }
    Ok(())
}

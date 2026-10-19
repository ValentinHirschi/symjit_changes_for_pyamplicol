//! `not` of a mask in complex kernels: the complexifier negated the (zero) imaginary part of a
//! mask kept in a complex slot too, so a negated mask was never zero for `iszero`, the test of
//! `if` (`append_if_else`) and kernel loops. Escape-time loops (`while n < limit && |z|^2 < 4`,
//! whose condition the kernel builder negates) stopped after one iteration with
//! `fast_complex = false`, in the bytecode interpreter and in code generated from the
//! complexified MIR.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{CompilerType, Composer, Slot, Translator};

/// out0 = if !(x < 1 && x < 4) { 2 } else { 3 }, out1 = !(x < 1 && x < 4) (a mask: all bits set
/// or zero): the `and` of two masks, which the direct translator keeps in stack slots, is complex
#[test]
fn negated_masks_in_complex_kernels() -> Result<()> {
    for ty in [CompilerType::Native, CompilerType::ByteCode] {
        for (direct, fast_complex, opt_level) in [
            (false, true, 1),
            (false, false, 1),
            (true, true, 1),
            (true, false, 0),
            (true, false, 1),
            (true, false, 2),
        ] {
            let mut config = base_config();
            config.set_opt_level(opt_level);
            config.set_complex(true);
            config.set_direct(direct);
            config.set_fast_complex(fast_complex);
            config.set_symbolica(true);
            config.ty = ty;

            let mut t = Translator::new(config);
            t.set_num_params(1);
            let one = t.append_constant(Complex::new(1.0, 0.0))?;
            let two = t.append_constant(Complex::new(2.0, 0.0))?;
            let three = t.append_constant(Complex::new(3.0, 0.0))?;
            let four = t.append_constant(Complex::new(4.0, 0.0))?;
            t.append_fun(
                &Slot::Temp(0),
                "lt",
                &[Slot::Param(0), Slot::Const(one)],
                false,
            )?;
            t.append_fun(
                &Slot::Temp(4),
                "lt",
                &[Slot::Param(0), Slot::Const(four)],
                false,
            )?;
            // a jump target: the masks are reloaded from their slots (as at the top of a
            // kernel loop), complex
            t.append_label(3)?;
            t.append_fun(
                &Slot::Temp(5),
                "and",
                &[Slot::Temp(0), Slot::Temp(4)],
                false,
            )?;
            t.append_fun(&Slot::Temp(1), "not", &[Slot::Temp(5)], false)?;
            t.append_if_else(&Slot::Temp(1), 5)?;
            t.append_assign(&Slot::Temp(2), &Slot::Const(two))?;
            t.append_goto(7)?;
            t.append_label(5)?;
            t.append_assign(&Slot::Temp(3), &Slot::Const(three))?;
            t.append_label(7)?;
            t.append_join(
                &Slot::Out(0),
                &Slot::Temp(1),
                &Slot::Temp(2),
                &Slot::Temp(3),
            )?;
            t.append_assign(&Slot::Out(1), &Slot::Temp(1))?;
            let app = t.compile()?;
            let mut outs = [Complex::default(); 2];
            let what = format!(
                "{ty:?}, direct = {direct}, fast_complex = {fast_complex}, opt level {opt_level}"
            );

            if ty == CompilerType::ByteCode {
                let mut app = app;
                for (x, then) in [(0.5, false), (1.5, true)] {
                    app.interpret(&[Complex::new(x, 0.25)], &mut outs);
                    check(&outs, then, &what);
                }
            } else {
                let app = app.seal()?;
                for (x, then) in [(0.5, false), (1.5, true)] {
                    app.evaluate(&[Complex::new(x, 0.25)], &mut outs);
                    check(&outs, then, &what);
                }
            }
        }
    }
    Ok(())
}

fn check(outs: &[Complex<f64>; 2], then: bool, what: &str) {
    assert_eq!(
        outs[0],
        Complex::new(if then { 2.0 } else { 3.0 }, 0.0),
        "{what}"
    );
    // the mask, real: its imaginary part is zero (the packed-complex kernels negate both lanes
    // of their registers; only the real part is a mask)
    assert_eq!(
        outs[1].re.to_bits(),
        if then { u64::MAX } else { 0 },
        "{what}"
    );
}

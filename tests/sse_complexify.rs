//! Complex kernels lowered by the complexifier (`fast_complex = false`) on the SSE generator,
//! which has no special cases there any more: it has the scalar AVX generator's complex
//! products and quotients (`times_complex`, `divide_complex`, with emulated fused operations)
//! and its `not` keeps the source register.
#![cfg(target_arch = "x86_64")]
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{CompilerType, Composer, Slot, Translator};

const NUM_OUTS: usize = 11;

fn build(ty: CompilerType, direct: bool, stable: bool, opt_level: u8) -> Result<Translator> {
    let mut config = base_config();
    config.ty = ty;
    config.set_opt_level(opt_level);
    config.set_complex(true);
    config.set_direct(direct);
    config.set_fast_complex(false);
    config.set_stable_math(stable);
    config.set_symbolica(true);

    let mut t = Translator::new(config);
    t.set_num_params(3);
    let (p0, p1, p2) = (Slot::Param(0), Slot::Param(1), Slot::Param(2));
    let two = Slot::Const(t.append_constant(Complex::new(2.0, 0.0))?);
    let one = Slot::Const(t.append_constant(Complex::new(1.0, 0.0))?);

    t.append_div(&Slot::Out(0), &p0, &p1, false, false)?;
    t.append_div(&Slot::Out(1), &two, &p1, true, false)?;
    t.append_fun(&Slot::Out(2), "recip", &[p0.clone()], false)?;
    t.append_mul(&Slot::Temp(0), &[p0.clone(), p1.clone()], 0)?;
    t.append_add(&Slot::Temp(0), &[Slot::Temp(0), p2.clone()], 0)?;
    t.append_fun(&Slot::Out(3), "root", &[Slot::Temp(0)], false)?;
    t.append_div(&Slot::Temp(1), &p0, &p1, false, false)?;
    t.append_fun(&Slot::Out(4), "abs", &[Slot::Temp(1)], false)?;
    t.append_fun(&Slot::Out(5), "sin", &[Slot::Temp(1)], false)?;
    t.append_pow(&Slot::Out(6), &p0, -3, false)?;
    // in place: t2 = p2 / t2 with t2 = p0 * p1
    t.append_mul(&Slot::Temp(2), &[p0.clone(), p1.clone()], 0)?;
    t.append_div(&Slot::Temp(2), &p2, &Slot::Temp(2), false, false)?;
    t.append_assign(&Slot::Out(7), &Slot::Temp(2))?;
    t.append_mul(&Slot::Out(8), &[Slot::Temp(1), p2.clone(), p0.clone()], 0)?;
    // a mask and its negation, both used afterwards
    t.append_fun(&Slot::Temp(3), "lt", &[p0.clone(), one], false)?;
    t.append_fun(&Slot::Out(9), "not", &[Slot::Temp(3)], false)?;
    t.append_assign(&Slot::Out(10), &Slot::Temp(3))?;
    Ok(t)
}

fn close(got: Complex<f64>, expected: Complex<f64>) -> bool {
    let part = |a: f64, b: f64| {
        a == b
            || (a.is_nan() && b.is_nan())
            // SSE emulates a fused multiply-add with two roundings
            || (a - b).abs() <= 1e-13 * expected.norm()
    };
    part(got.re, expected.re) && part(got.im, expected.im)
}

#[test]
fn sse_complexified_kernels_match_bytecode() -> Result<()> {
    if cfg!(not(target_arch = "x86_64")) {
        return Ok(());
    }

    let points = [
        [
            Complex::new(0.5, 0.25),
            Complex::new(-1.5, 2.0),
            Complex::new(0.75, -3.0),
        ],
        [
            Complex::new(2.5, -1.0),
            Complex::new(0.125, 0.5),
            Complex::new(-2.0, 1.0),
        ],
        [
            Complex::new(1e300, 1e300),
            Complex::new(3e299, -2e300),
            Complex::new(1e-300, 1e-310),
        ],
        [
            Complex::new(1e-310, 2e-310),
            Complex::new(0.0, 3e-310),
            Complex::new(1.0, 5e-320),
        ],
    ];

    for direct in [false, true] {
        for stable in [false, true] {
            for opt_level in 0..=3 {
                let what = format!("direct = {direct}, stable = {stable}, opt level {opt_level}");
                let sse = build(CompilerType::AmdSSE, direct, stable, opt_level)?
                    .compile()?
                    .seal()?;
                let mut bytecode =
                    build(CompilerType::ByteCode, direct, stable, opt_level)?.compile()?;

                for p in points.iter() {
                    let mut got = [Complex::default(); NUM_OUTS];
                    let mut expected = [Complex::default(); NUM_OUTS];
                    sse.evaluate(p, &mut got);
                    bytecode.interpret(p, &mut expected);
                    for (k, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                        assert!(close(*g, *e), "out {k}: {g} != {e} at {p:?}, {what}");
                    }
                    // the mask and its negation (real; all bits or zero)
                    let mask = if p[0].re < 1.0 { u64::MAX } else { 0 };
                    assert_eq!(got[10].re.to_bits(), mask, "{what}");
                    assert_eq!(got[9].re.to_bits(), !mask, "{what}");
                }
            }
        }
    }
    Ok(())
}

/// Real kernels at every opt level: the register allocator gives the SSE generator the same
/// registers as the others (a destination may be one of the sources), so the non-commutative
/// operations, comparisons and `ifelse` must not modify their sources.
#[test]
fn sse_real_kernels_match_bytecode() -> Result<()> {
    if cfg!(not(target_arch = "x86_64")) {
        return Ok(());
    }

    let build = |ty: CompilerType, opt_level: u8| -> Result<Translator> {
        let mut config = base_config();
        config.ty = ty;
        config.set_opt_level(opt_level);
        config.set_symbolica(true);

        let mut t = Translator::new(config);
        t.set_num_params(3);
        let (p0, p1, p2) = (Slot::Param(0), Slot::Param(1), Slot::Param(2));
        let three = Slot::Const(t.append_constant(Complex::new(3.0, 0.0))?);
        let r = |i| Slot::Temp(i);

        // t0 = p1 + p0, t1 = t0 / p2, t2 = p1 / t1
        t.append_add(&r(0), &[p1.clone(), p0.clone()], 0)?;
        t.append_div(&r(1), &r(0), &p2, true, true)?;
        t.append_div(&r(2), &p1, &r(1), true, true)?;
        t.append_assign(&Slot::Out(0), &r(2))?;
        t.append_fun(&r(3), "lt", &[r(1), p0.clone()], true)?;
        t.append_fun(&r(4), "geq", &[p2.clone(), r(2)], true)?;
        t.append_fun(&r(5), "and", &[r(3), r(4)], true)?;
        t.append_fun(&r(6), "not", &[r(5)], true)?;
        t.append_assign(&Slot::Out(1), &r(5))?;
        t.append_assign(&Slot::Out(2), &r(6))?;
        // out3 = if r5 { t2 } else { t1 }, out4 = if r6 { 3 } else { t1 * p0 }
        t.append_if_else(&r(5), 1)?;
        t.append_assign(&r(7), &r(2))?;
        t.append_goto(2)?;
        t.append_label(1)?;
        t.append_assign(&r(8), &r(1))?;
        t.append_label(2)?;
        t.append_join(&Slot::Out(3), &r(5), &r(7), &r(8))?;
        t.append_mul(&r(9), &[r(1), p0.clone()], 0)?;
        t.append_if_else(&r(6), 3)?;
        t.append_assign(&r(10), &three)?;
        t.append_goto(4)?;
        t.append_label(3)?;
        t.append_assign(&r(11), &r(9))?;
        t.append_label(4)?;
        t.append_join(&Slot::Out(4), &r(6), &r(10), &r(11))?;
        t.append_fun(&Slot::Out(5), "gt", &[r(9), r(0)], true)?;
        t.append_fun(&Slot::Out(6), "leq", &[r(0), r(9)], true)?;
        Ok(t)
    };

    let points = [
        [0.5, 0.25, -1.5],
        [2.0, -3.0, 0.75],
        [-1.0, 4.0, 2.5],
        [3.0, 3.0, 0.125],
    ];
    for opt_level in 0..=3 {
        let sse = build(CompilerType::AmdSSE, opt_level)?.compile()?.seal()?;
        let mut bytecode = build(CompilerType::ByteCode, opt_level)?.compile()?;
        for p in points.iter() {
            let mut got = [0.0; 7];
            let mut expected = [0.0; 7];
            sse.evaluate(p, &mut got);
            bytecode.interpret(p, &mut expected);
            for (k, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "out {k}: {g} != {e} at {p:?}, opt level {opt_level}"
                );
            }
        }
    }
    Ok(())
}

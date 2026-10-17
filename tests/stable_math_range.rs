// Range checks of `Config::stable_math` against exact rational references, on a grid of
// magnitudes from the smallest subnormal to f64::MAX (both translators, fast_complex on and
// off, scalar and AVX lanes).
mod common;

use common::base_config;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use symjit::{Applet, Complex, Composer, Slot, Translator};

const MAGNITUDES: [f64; 10] = [
    0.,
    5e-324,
    3.3e-310,
    2.3e-200,
    0.75,
    3.,
    6.1e160,
    1.3e300,
    9e307,
    f64::MAX,
];

/// Results above this bound may overflow spuriously (see `known_division_corner_cases`).
const SAFE_MAX: f64 = f64::MAX / 4.;

fn grid(magnitudes: &[f64]) -> Vec<Complex<f64>> {
    let mut grid = Vec::new();
    for (i, &re) in magnitudes.iter().enumerate() {
        for (j, &im) in magnitudes.iter().enumerate() {
            let signs = (i * 7 + j * 3) % 4;
            grid.push(Complex::new(
                if signs & 1 == 0 { re } else { -re },
                if signs & 2 == 0 { im } else { -im },
            ));
        }
    }
    grid
}

fn rational(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

fn rounded(x: &BigRational) -> f64 {
    x.to_f64().unwrap_or(if x.is_negative() {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    })
}

/// `x` times 2^1074, an integer for every finite `x`: integer-valued rationals keep the
/// exact arithmetic below cheap.
fn integer(x: f64) -> BigRational {
    let scale = rational(2f64.powi(537));
    rational(x) * &scale * &scale
}

/// n / d rounded, for integer-valued n and d > 0 (left unreduced).
fn quotient(n: &BigRational, d: &BigRational) -> f64 {
    rounded(&BigRational::new_raw(n.to_integer(), d.to_integer()))
}

/// x / y rounded componentwise; None for y = 0.
fn exact_div(x: Complex<f64>, y: Complex<f64>) -> Option<Complex<f64>> {
    let (a, b, c, d) = (integer(x.re), integer(x.im), integer(y.re), integer(y.im));
    let denominator = &c * &c + &d * &d;
    if denominator.is_zero() {
        return None;
    }
    Some(Complex::new(
        quotient(&(&a * &c + &b * &d), &denominator),
        quotient(&(&b * &c - &a * &d), &denominator),
    ))
}

/// x^-n rounded componentwise; None for x = 0.
fn exact_inverse_power(x: Complex<f64>, n: u32) -> Option<Complex<f64>> {
    // x = (a + ib) / K with K = 2^1074, so x^-n = K^n conj(w) / |w|^2 for w = (a + ib)^n
    let (a, b) = (integer(x.re), integer(x.im));
    let (mut re, mut im) = (BigRational::from_integer(1.into()), BigRational::zero());
    let mut scale = BigRational::from_integer(1.into());
    for _ in 0..n {
        (re, im) = (&re * &a - &im * &b, &re * &b + &im * &a);
        scale *= integer(1.);
    }
    let denominator = &re * &re + &im * &im;
    if denominator.is_zero() {
        return None;
    }
    Some(Complex::new(
        quotient(&(&scale * &re), &denominator),
        quotient(&(-(&scale * &im)), &denominator),
    ))
}

fn ulp(x: f64) -> f64 {
    let x = x.abs();
    if x < f64::MIN_POSITIVE {
        f64::from_bits(1)
    } else {
        f64::from_bits(x.to_bits() + 1) - x
    }
}

/// The normwise error of `actual` in units of the last place of the larger component of
/// `expected` (subnormal results are measured in units of the smallest subnormal).
fn error_units(actual: Complex<f64>, expected: Complex<f64>) -> f64 {
    let unit = ulp(expected.re.abs().max(expected.im.abs()));
    let error = (actual.re - expected.re).hypot(actual.im - expected.im) / unit;
    if error.is_nan() {
        f64::INFINITY
    } else {
        error
    }
}

fn in_safe_range(z: Complex<f64>) -> bool {
    z.re.is_finite() && z.im.is_finite() && z.re.abs().max(z.im.abs()) <= SAFE_MAX
}

/// Evaluates `app` at every point of `inputs`, on the scalar kernel and, if available,
/// on the AVX kernel with one point per lane.
fn evaluate_all(
    app: &Applet,
    inputs: &[Vec<Complex<f64>>],
) -> Vec<(&'static str, Vec<Complex<f64>>)> {
    let mut results = vec![(
        "scalar",
        inputs
            .iter()
            .map(|args| {
                let mut out = [Complex::default()];
                app.evaluate(args, &mut out);
                out[0]
            })
            .collect(),
    )];
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") && app.compiled_simd.is_some() {
        use wide::f64x4;
        let n = inputs[0].len();
        let mut simd = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(4) {
            let lane = |k: usize| &chunk[k.min(chunk.len() - 1)];
            let args: Vec<Complex<f64x4>> = (0..n)
                .map(|p| {
                    Complex::new(
                        f64x4::from([lane(0)[p].re, lane(1)[p].re, lane(2)[p].re, lane(3)[p].re]),
                        f64x4::from([lane(0)[p].im, lane(1)[p].im, lane(2)[p].im, lane(3)[p].im]),
                    )
                })
                .collect();
            let mut out = [Complex::new(f64x4::ZERO, f64x4::ZERO)];
            app.evaluate(&args, &mut out);
            let (re, im) = (out[0].re.to_array(), out[0].im.to_array());
            for k in 0..chunk.len() {
                simd.push(Complex::new(re[k], im[k]));
            }
        }
        results.push(("simd", simd));
    }
    results
}

enum Op {
    Div,
    Recip,
    Power(i64),
    Abs,
    Root,
}

fn compile(op: &Op, direct: bool, fast_complex: bool, stable_math: bool) -> anyhow::Result<Applet> {
    let mut config = base_config();
    config.set_direct(direct);
    config.set_complex(true);
    config.set_fast_complex(fast_complex);
    config.set_stable_math(stable_math);
    let mut translator = Translator::new(config);
    let (out, x, y) = (Slot::Out(0), Slot::Param(0), Slot::Param(1));
    match op {
        Op::Div => {
            translator.set_num_params(2);
            translator.append_div(&out, &x, &y, false, false)?;
        }
        Op::Recip => {
            translator.set_num_params(1);
            translator.append_fun(&out, "recip", &[x], false)?;
        }
        Op::Power(p) => {
            translator.set_num_params(1);
            translator.append_pow(&out, &x, *p, false)?;
        }
        Op::Abs => {
            translator.set_num_params(1);
            translator.append_fun(&out, "abs", &[x], false)?;
        }
        Op::Root => {
            translator.set_num_params(1);
            translator.append_fun(&out, "root", &[x], false)?;
        }
    }
    translator.compile()?.seal()
}

/// Checks `op` over `inputs` against `reference` (None: not checked) with `tolerance` units
/// (see `error_units`) in every configuration; returns the failures.
fn check(
    name: &str,
    op: Op,
    inputs: &[Vec<Complex<f64>>],
    reference: impl Fn(&[Complex<f64>]) -> Option<Complex<f64>>,
    tolerance: f64,
) -> anyhow::Result<Vec<String>> {
    let expected: Vec<Option<Complex<f64>>> = inputs.iter().map(|args| reference(args)).collect();
    let mut failures = Vec::new();
    for direct in [false, true] {
        for fast_complex in [false, true] {
            let app = compile(&op, direct, fast_complex, true)?;
            for (mode, results) in evaluate_all(&app, inputs) {
                for ((args, actual), expected) in inputs.iter().zip(results).zip(&expected) {
                    let Some(expected) = *expected else { continue };
                    if !in_safe_range(expected) {
                        continue;
                    }
                    let error = error_units(actual, expected);
                    if error > tolerance && failures.len() < 20 {
                        failures.push(format!(
                            "{name} {args:?} (direct={direct}, fast_complex={fast_complex}, {mode}): {actual:?} != {expected:?} ({error:.1} units)"
                        ));
                    }
                }
            }
        }
    }
    Ok(failures)
}

#[test]
fn stable_division_matches_exact_quotients_below_overflow() -> anyhow::Result<()> {
    let points = grid(&MAGNITUDES);
    let mut inputs = Vec::new();
    for &x in points.iter() {
        for &y in points.iter() {
            inputs.push(vec![x, y]);
        }
    }
    let failures = check("div", Op::Div, &inputs, |a| exact_div(a[0], a[1]), 4.)?;
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn stable_reciprocal_and_inverse_powers_match_exact_values() -> anyhow::Result<()> {
    let inputs: Vec<Vec<Complex<f64>>> = grid(&MAGNITUDES).into_iter().map(|z| vec![z]).collect();
    let mut failures = check(
        "recip",
        Op::Recip,
        &inputs,
        |a| exact_div(Complex::new(1., 0.), a[0]),
        4.,
    )?;
    for p in [-1, -2, -3] {
        failures.extend(check(
            &format!("pow{p}"),
            Op::Power(p),
            &inputs,
            |a| exact_inverse_power(a[0], p.unsigned_abs() as u32),
            4. * p.unsigned_abs() as f64,
        )?);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn stable_abs_and_root_are_accurate_across_the_range() -> anyhow::Result<()> {
    let inputs: Vec<Vec<Complex<f64>>> = grid(&MAGNITUDES).into_iter().map(|z| vec![z]).collect();
    let mut failures = Vec::new();
    for direct in [false, true] {
        for fast_complex in [false, true] {
            let abs = compile(&Op::Abs, direct, fast_complex, true)?;
            let root = compile(&Op::Root, direct, fast_complex, true)?;
            for ((mode, abs), (_, root)) in evaluate_all(&abs, &inputs)
                .into_iter()
                .zip(evaluate_all(&root, &inputs))
            {
                for ((args, abs), root) in inputs.iter().zip(abs).zip(root) {
                    let z = args[0];
                    let (a, b) = (rational(z.re), rational(z.im));
                    let norm = &a * &a + &b * &b;
                    let context =
                        format!("{z:?} (direct={direct}, fast_complex={fast_complex}, {mode})");
                    // |z| (below SAFE_MAX) lies within 2 ulp of the result
                    if norm <= rational(SAFE_MAX) * rational(SAFE_MAX) {
                        let ok = abs.re.is_finite() && abs.im == 0. && {
                            let u = 2. * ulp(abs.re);
                            let lower = rational((abs.re - u).max(0.));
                            let upper = rational(abs.re) + rational(u);
                            &lower * &lower <= norm && norm <= &upper * &upper
                        };
                        if !ok {
                            failures.push(format!("abs {context}: {abs:?}"));
                        }
                    }
                    // sqrt(z): principal branch and r^2 = z to a few ulp
                    let (r, s) = (rational(root.re), rational(root.im));
                    let residual = {
                        let re = &r * &r - &s * &s - &a;
                        let im = (&r * &s) * BigRational::from_integer(2.into()) - &b;
                        &re * &re + &im * &im
                    };
                    let error = if norm.is_zero() {
                        if root.re == 0. && root.im == 0. {
                            0.
                        } else {
                            f64::INFINITY
                        }
                    } else {
                        rounded(&(residual / &norm)).sqrt() / f64::EPSILON
                    };
                    let branch = root.re >= 0.
                        && (z.im == 0. || root.im.signum() == z.im.signum() || root.im == 0.);
                    if !root.re.is_finite() || !root.im.is_finite() || !branch || error > 8. {
                        failures.push(format!(
                            "root {context}: {root:?} (|r^2 - z| / |z| = {error:.1} eps)"
                        ));
                    }
                }
            }
        }
    }
    failures.truncate(20);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn stable_math_keeps_default_bits_in_the_normal_range() -> anyhow::Result<()> {
    let points = grid(&[0., 1e-50, 0.75, 3., 1e50]);
    let unary: Vec<Vec<Complex<f64>>> = points.iter().map(|&z| vec![z]).collect();
    let mut binary = Vec::new();
    for &x in points.iter() {
        for &y in points.iter() {
            binary.push(vec![x, y]);
        }
    }
    let mut failures = Vec::new();
    for (name, op, inputs) in [
        ("div", Op::Div, &binary),
        ("abs", Op::Abs, &unary),
        ("root", Op::Root, &unary),
    ] {
        for direct in [false, true] {
            for fast_complex in [false, true] {
                let stable = evaluate_all(&compile(&op, direct, fast_complex, true)?, inputs);
                let default = evaluate_all(&compile(&op, direct, fast_complex, false)?, inputs);
                for ((mode, stable), (_, default)) in stable.into_iter().zip(default) {
                    for ((args, s), d) in inputs.iter().zip(stable).zip(default) {
                        let same = |x: f64, y: f64| {
                            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
                        };
                        if !same(s.re, d.re) || !same(s.im, d.im) {
                            failures.push(format!("{name} {args:?} (direct={direct}, fast_complex={fast_complex}, {mode}): stable {s:?} != default {d:?}"));
                        }
                    }
                }
            }
        }
    }
    failures.truncate(20);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

/// Division cases where the scaled formula of `stable_math` differs from the exact quotient:
/// the numerator is scaled by the factor of the denominator, so it can overflow when |x / y|
/// is within a factor of about 4 of f64::MAX; an overflowing component turns its finite
/// companion into NaN; and when the scale factor is below 1 (|y| above about 2), a subnormal
/// component of the denominator loses its last bits, and with them a normal component of the
/// quotient.
#[test]
#[ignore = "known stable_math division corner cases"]
fn known_division_corner_cases() -> anyhow::Result<()> {
    let cases = [
        // spurious overflow: |x / y| = 1.2e308
        (Complex::new(0., 9e307), Complex::new(-0., -0.75)),
        // the real part overflows; the imaginary part is exactly 0
        (Complex::new(0., f64::MAX), Complex::new(0., 5e-324)),
        // the imaginary part 2.45e-307 is normal
        (Complex::new(0., 3.1e17), Complex::new(5e-324, 2.5)),
    ];
    let inputs: Vec<Vec<Complex<f64>>> = cases.iter().map(|&(x, y)| vec![x, y]).collect();
    let mut failures = Vec::new();
    for direct in [false, true] {
        for fast_complex in [false, true] {
            let app = compile(&Op::Div, direct, fast_complex, true)?;
            for (mode, results) in evaluate_all(&app, &inputs) {
                for (&(x, y), actual) in cases.iter().zip(results) {
                    let expected = exact_div(x, y).unwrap();
                    let same =
                        |a: f64, e: f64| a == e || (e.is_finite() && (a - e).abs() <= ulp(e));
                    if !same(actual.re, expected.re) || !same(actual.im, expected.im) {
                        failures.push(format!("({x:?}) / ({y:?}) (direct={direct}, fast_complex={fast_complex}, {mode}): {actual:?} != {expected:?}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

use anyhow::{anyhow, Result};
use num_complex::Complex;
use spec_math::cephes64;
use std::ffi::c_void;
use std::fmt;

use super::applet::Applet;

/*
 * Design Note:
 *
 * Unary complex functions are defined as:
 *      `extern "C" fn(f64, f64, &mut Complex<f64>)`
 *
 * Initially, we tried `extern "C" fn(Complex<f64>) -> Complex<f64>`.
 * However, Complex is a user-defined structure in Rust and not an
 * intrinsic type. Therefore, the C-mapping is variable. This worked
 * fine on Linux systems but not Windows that passes structures in
 * stack and not registers.
 *
 * Next, we tried `fn(Complex<f64>) -> Complex<f64>`. In practice, this
 * works but has a main shortcoming. It depends on the details of
 * Rust ABI, which is not stable. In fact, this only works when the
 * crate is compiled in release mode and not debug mode.
 *
 * Therefore, we settled on the current version, where the function
 * arguments are defined explicitely and the C-mapping is unambiguous.
 * This version has the additional benefit of allowing call elision
 * (not implemented yet).
 *
 */

#[repr(C)]
pub struct SinCos {
    pub s: f64,
    pub c: f64,
}

pub type UnaryFunc = extern "C" fn(f64) -> f64;
pub type BinaryFunc = extern "C" fn(f64, f64) -> f64;
pub type UnaryFuncCplx = extern "C" fn(f64, f64, &mut Complex<f64>);
pub type BinaryFuncCplx = extern "C" fn(f64, f64, &mut Complex<f64>);
pub type PairedUnaryFunc = extern "C" fn(f64) -> SinCos;

#[derive(Clone)]
pub enum Func {
    Unary(UnaryFunc),
    Binary(BinaryFunc),
    UnaryCplx(UnaryFuncCplx),
    BinaryCplx(BinaryFuncCplx),
    PairedUnary(PairedUnaryFunc),
    Slice {
        f_scalar: *const c_void,
        f_simd: *const c_void,
        env: *const c_void,
    },
    App(Box<Applet>),
    Recursive,
}

impl Func {
    pub fn func_ptr(&self) -> u64 {
        match self {
            Func::Unary(f) => *f as usize as u64,
            Func::Binary(f) => *f as usize as u64,
            Func::UnaryCplx(f) => *f as usize as u64,
            Func::BinaryCplx(f) => *f as usize as u64,
            Func::PairedUnary(f) => *f as usize as u64,
            Func::Slice { .. } => 0,
            Func::App(app) => match app.scalar_ptr() {
                Some(f) => f as usize as u64,
                None => 0,
            },
            Func::Recursive => 0,
        }
    }
}

unsafe impl Sync for Func {}
unsafe impl Send for Func {}

impl fmt::Debug for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<function pointer>")
    }
}

// std already links libm; an explicit #[link(name = "m")] moves it ahead of
// compiler_builtins and silently replaces the real cbrt with glibc's (~3 ulp).
#[cfg(all(target_family = "unix", feature = "libm"))]
extern "C" {
    fn csin(z: Complex<f64>) -> Complex<f64>;
    fn ccos(z: Complex<f64>) -> Complex<f64>;
    fn ctan(z: Complex<f64>) -> Complex<f64>;
    fn casin(z: Complex<f64>) -> Complex<f64>;
    fn cacos(z: Complex<f64>) -> Complex<f64>;
    fn catan(z: Complex<f64>) -> Complex<f64>;
    fn csinh(z: Complex<f64>) -> Complex<f64>;
    fn ccosh(z: Complex<f64>) -> Complex<f64>;
    fn ctanh(z: Complex<f64>) -> Complex<f64>;
    fn casinh(z: Complex<f64>) -> Complex<f64>;
    fn cacosh(z: Complex<f64>) -> Complex<f64>;
    fn catanh(z: Complex<f64>) -> Complex<f64>;
}

// 1/w for the reciprocal functions (csc, sec, cot, csch, sech, coth). `Complex::inv`
// computes conj(w)/|w|^2, which overflows for |w| > 1.3e154 (giving 0 instead of tiny
// values) and gives NaN for an infinite w. Smith's algorithm avoids the overflow, and
// 1/inf = 0 with the signs of conj(1/w) (C99 Annex G); NaN arguments stay NaN.
fn recip(w: Complex<f64>) -> Complex<f64> {
    if w.re.is_infinite() || w.im.is_infinite() {
        return Complex::new(0.0f64.copysign(w.re), (-0.0f64).copysign(w.im));
    }
    if w.re.abs() >= w.im.abs() {
        let r = w.im / w.re;
        let d = w.re + w.im * r;
        Complex::new(1.0 / d, -r / d)
    } else {
        let r = w.re / w.im;
        let d = w.re * r + w.im;
        Complex::new(r / d, -1.0 / d)
    }
}

// Keep the exponent separate while forming complex products. Scaling an entire
// complex operand by its largest component can erase a small companion before
// division brings it back into range.
fn finite_parts(value: f64) -> (f64, i32) {
    if value == 0.0 {
        return (value, 0);
    }
    let mut value = value;
    let mut adjustment = 0;
    if value.is_subnormal() {
        value *= (1u64 << 54) as f64;
        adjustment = -54;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + adjustment;
    let mantissa = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | (1023u64 << 52));
    (mantissa, exponent)
}

fn scale_binary(mut value: f64, mut exponent: i32) -> f64 {
    // Multiplications by normal powers of two are exact until the final
    // rounding. Leave 53 bits of headroom before producing subnormal results.
    while exponent > 1023 {
        value *= f64::from_bits(2046u64 << 52);
        exponent -= 1023;
    }
    while exponent < -1022 {
        value *= f64::from_bits(54u64 << 52); // 2^(-1022+53)
        exponent += 969;
    }
    value * f64::from_bits(((exponent + 1023) as u64) << 52)
}

fn finite_product_sum(a: f64, b: f64, c: f64, d: f64) -> (f64, i32) {
    let (a, ea) = finite_parts(a);
    let (b, eb) = finite_parts(b);
    let (c, ec) = finite_parts(c);
    let (d, ed) = finite_parts(d);
    let left = a * b;
    let right = c * d;
    if left == 0.0 && right == 0.0 {
        return (left + right, 0);
    }
    if left == 0.0 {
        return (right, ec + ed);
    }
    if right == 0.0 {
        return (left, ea + eb);
    }
    let exponent = (ea + eb).max(ec + ed);
    (
        scale_binary(left, ea + eb - exponent) + scale_binary(right, ec + ed - exponent),
        exponent,
    )
}

fn divide_complex(x: Complex<f64>, y: Complex<f64>) -> Complex<f64> {
    // These branches also retain signed zeros and independently finite
    // components when the other quotient component genuinely overflows.
    if y.im == 0.0 {
        return Complex::new(x.re / y.re, x.im / y.re);
    }
    if y.re == 0.0 {
        return Complex::new(x.im / y.im, -x.re / y.im);
    }
    if ![x.re, x.im, y.re, y.im].into_iter().all(f64::is_finite) {
        return x / y;
    }
    let lower = f64::from_bits(573u64 << 52); // 2^-450
    let upper = f64::from_bits(1473u64 << 52); // 2^450
    if [x.re, x.im, y.re, y.im]
        .into_iter()
        .all(|value| value == 0.0 || (lower..=upper).contains(&value.abs()))
    {
        return x / y;
    }
    let (denominator, denominator_exponent) = finite_product_sum(y.re, y.re, y.im, y.im);
    let quotient = |a, b, c, d| {
        let (numerator, exponent) = finite_product_sum(a, b, c, d);
        scale_binary(numerator / denominator, exponent - denominator_exponent)
    };
    Complex::new(
        quotient(x.re, y.re, x.im, y.im),
        quotient(x.im, y.re, -x.re, y.im),
    )
}

// tanh for the default (non-libm) build. `Complex::tanh` divides sinh(2x) by
// cosh(2x) + cos(2y), which overflows to NaN for |x| > ~355 and cancels near the poles
// z = i(k + 1/2)pi, where it lost up to 1e8 ulp (and hundreds of ulp for |x|, |y| < 3).
// Instead, tanh z = sinh z conj(cosh z) / |cosh z|^2:
//
//   tanh z = (sinh x cosh x + i sin y cos y) / (sinh^2 x + cos^2 y),
//
// whose denominator is a sum of squares (no cancellation; the same formula as `libm::ctanh`).
// For |x| > 22, tanh(z) is +-1 to double precision and its imaginary part is
// 4 sin(y) cos(y) exp(-2|x|) (as in C99 ctanh).
#[allow(dead_code)] // unused with the `libm` feature
fn tanh_c(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    let (sy, cy) = y.sin_cos();
    if x.abs() > 22.0 {
        let e = (-2.0 * x.abs()).exp();
        Complex::new(1.0f64.copysign(x), 4.0 * sy * cy * e)
    } else {
        let (s, c) = (x.sinh(), x.cosh());
        let d = s * s + cy * cy;
        Complex::new(s * c / d, sy * cy / d)
    }
}

// tan(z) = -i tanh(iz)
#[allow(dead_code)] // unused with the `libm` feature
fn tan_c(z: Complex<f64>) -> Complex<f64> {
    let t = tanh_c(Complex::new(-z.im, z.re));
    Complex::new(t.im, -t.re)
}

pub struct VirtualTable;

impl VirtualTable {
    /// Names used by intrinsic lowering cannot be rebound to user callbacks.
    pub(crate) fn is_internal_name(name: &str) -> bool {
        name.strip_prefix("cplx_")
            .unwrap_or(name)
            .starts_with("__symjit_")
    }
    /// The C math library (libm) function computing `op`, for object files; None if
    /// there is none (e.g. `csc`, `sinc`, complex functions). Note that symjit's `log`
    /// is the decimal logarithm and `ln` the natural one.
    pub fn c_name(op: &str) -> Option<&'static str> {
        let name = match op {
            "sin" => "sin",
            "cos" => "cos",
            "tan" => "tan",
            "sinh" => "sinh",
            "cosh" => "cosh",
            "tanh" => "tanh",
            "arcsin" => "asin",
            "arccos" => "acos",
            "arctan" => "atan",
            "arcsinh" => "asinh",
            "arccosh" => "acosh",
            "arctanh" => "atanh",
            "exp" => "exp",
            "expm1" => "expm1",
            "exp2" => "exp2",
            "ln" => "log",
            "log" => "log10",
            "log1p" => "log1p",
            "log2" => "log2",
            "cbrt" => "cbrt",
            "erf" => "erf",
            "erfc" => "erfc",
            "gamma" => "tgamma",
            "loggamma" => "lgamma",
            "power" => "pow",
            "atan2" => "atan2",
            _ => return None,
        };
        Some(name)
    }

    // Finds the function reference for op
    pub fn from_str(op: &str) -> Result<Func> {
        let f = match op {
            "sin" => Func::Unary(Self::sin),
            "sinc" => Func::Unary(Self::sinc),
            "cos" => Func::Unary(Self::cos),
            "sin_cos" => Func::PairedUnary(Self::sin_cos),
            "tan" => Func::Unary(Self::tan),
            "csc" => Func::Unary(Self::csc),
            "sec" => Func::Unary(Self::sec),
            "cot" => Func::Unary(Self::cot),
            "sinh" => Func::Unary(Self::sinh),
            "cosh" => Func::Unary(Self::cosh),
            "tanh" => Func::Unary(Self::tanh),
            "csch" => Func::Unary(Self::csch),
            "sech" => Func::Unary(Self::sech),
            "coth" => Func::Unary(Self::coth),
            "arcsin" => Func::Unary(Self::asin),
            "arccos" => Func::Unary(Self::acos),
            "arctan" => Func::Unary(Self::atan),
            "arcsinh" => Func::Unary(Self::asinh),
            "arccosh" => Func::Unary(Self::acosh),
            "arctanh" => Func::Unary(Self::atanh),
            "cbrt" => Func::Unary(Self::cbrt),
            "exp" => Func::Unary(Self::exp),
            "ln" => Func::Unary(Self::ln),
            "log" => Func::Unary(Self::log),
            "expm1" => Func::Unary(Self::expm1),
            "log1p" => Func::Unary(Self::log1p),
            "exp2" => Func::Unary(Self::exp2),
            "log2" => Func::Unary(Self::log2),
            "erf" => Func::Unary(Self::erf),
            "erfc" => Func::Unary(Self::erfc),
            "gamma" => Func::Unary(Self::gamma),
            "loggamma" => Func::Unary(Self::loggamma),
            "Si" => Func::Unary(Self::si),
            "Ci" => Func::Unary(Self::ci),
            "Shi" => Func::Unary(Self::shi),
            "Chi" => Func::Unary(Self::chi),
            "random" => Func::Unary(Self::random),
            // Binary Functions
            "power" => Func::Binary(Self::power),
            "atan2" => Func::Binary(Self::atan2),
            "arg" => Func::Unary(Self::arg),
            "__symjit_abs" => Func::Unary(Self::stable_abs),
            "__symjit_divide" => Func::Binary(Self::stable_divide),
            "__symjit_root" => Func::Unary(Self::stable_root),
            // Unary Complex Functions
            "cplx___symjit_abs" => Func::UnaryCplx(Self::cplx_stable_abs),
            "cplx___symjit_root" => Func::UnaryCplx(Self::cplx_root),
            "cplx_sin" => Func::UnaryCplx(Self::cplx_sin),
            "cplx_sinc" => Func::UnaryCplx(Self::cplx_sinc),
            "cplx_cos" => Func::UnaryCplx(Self::cplx_cos),
            "cplx_tan" => Func::UnaryCplx(Self::cplx_tan),
            "cplx_csc" => Func::UnaryCplx(Self::cplx_csc),
            "cplx_sec" => Func::UnaryCplx(Self::cplx_sec),
            "cplx_cot" => Func::UnaryCplx(Self::cplx_cot),
            "cplx_sinh" => Func::UnaryCplx(Self::cplx_sinh),
            "cplx_cosh" => Func::UnaryCplx(Self::cplx_cosh),
            "cplx_tanh" => Func::UnaryCplx(Self::cplx_tanh),
            "cplx_csch" => Func::UnaryCplx(Self::cplx_csch),
            "cplx_sech" => Func::UnaryCplx(Self::cplx_sech),
            "cplx_coth" => Func::UnaryCplx(Self::cplx_coth),
            "cplx_arcsin" => Func::UnaryCplx(Self::cplx_asin),
            "cplx_arccos" => Func::UnaryCplx(Self::cplx_acos),
            "cplx_arctan" => Func::UnaryCplx(Self::cplx_atan),
            "cplx_arcsinh" => Func::UnaryCplx(Self::cplx_asinh),
            "cplx_arccosh" => Func::UnaryCplx(Self::cplx_acosh),
            "cplx_arctanh" => Func::UnaryCplx(Self::cplx_atanh),
            "cplx_cbrt" => Func::UnaryCplx(Self::cplx_cbrt),
            "cplx_exp" => Func::UnaryCplx(Self::cplx_exp),
            "cplx_ln" => Func::UnaryCplx(Self::cplx_ln),
            "cplx_log" => Func::UnaryCplx(Self::cplx_log),
            // Complex Binary Functions
            "cplx___symjit_divide" => Func::BinaryCplx(Self::cplx_stable_divide),
            "cplx_power" => Func::BinaryCplx(Self::cplx_power),
            "cplx_random" => Func::UnaryCplx(Self::cplx_random),
            "cplx_arg" => Func::UnaryCplx(Self::cplx_arg),
            // yield points of lockstep evaluation (`lockstep.rs`)
            #[cfg(feature = "async")]
            "_yield_" => Func::Unary(super::lockstep::yield_point),
            #[cfg(feature = "async")]
            "cplx__yield_" => Func::UnaryCplx(super::lockstep::cplx_yield_point),
            _ => {
                return Err(anyhow!("op_code {} is not found or is not supported", op));
            }
        };

        Ok(f)
    }

    pub extern "C" fn power(x: f64, y: f64) -> f64 {
        x.powf(y)
    }

    pub extern "C" fn atan2(x: f64, y: f64) -> f64 {
        x.atan2(y)
    }

    pub extern "C" fn sinc(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            x.sin() / x
        }
    }

    pub extern "C" fn sin(x: f64) -> f64 {
        x.sin()
    }

    pub extern "C" fn cos(x: f64) -> f64 {
        x.cos()
    }

    pub extern "C" fn sin_cos(x: f64) -> SinCos {
        let (s, c) = x.sin_cos();
        SinCos { s, c }
    }

    pub extern "C" fn tan(x: f64) -> f64 {
        x.tan()
    }

    pub extern "C" fn csc(x: f64) -> f64 {
        1.0 / x.sin()
    }

    pub extern "C" fn sec(x: f64) -> f64 {
        1.0 / x.cos()
    }

    pub extern "C" fn cot(x: f64) -> f64 {
        1.0 / x.tan()
    }

    pub extern "C" fn sinh(x: f64) -> f64 {
        x.sinh()
    }

    pub extern "C" fn cosh(x: f64) -> f64 {
        x.cosh()
    }

    pub extern "C" fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    pub extern "C" fn csch(x: f64) -> f64 {
        1.0 / x.sinh()
    }

    pub extern "C" fn sech(x: f64) -> f64 {
        1.0 / x.cosh()
    }

    pub extern "C" fn coth(x: f64) -> f64 {
        1.0 / x.tanh()
    }

    pub extern "C" fn asin(x: f64) -> f64 {
        x.asin()
    }

    pub extern "C" fn acos(x: f64) -> f64 {
        x.acos()
    }

    pub extern "C" fn atan(x: f64) -> f64 {
        x.atan()
    }

    pub extern "C" fn asinh(x: f64) -> f64 {
        x.asinh()
    }

    pub extern "C" fn acosh(x: f64) -> f64 {
        x.acosh()
    }

    pub extern "C" fn atanh(x: f64) -> f64 {
        x.atanh()
    }

    pub extern "C" fn cbrt(x: f64) -> f64 {
        x.cbrt()
    }

    pub extern "C" fn exp(x: f64) -> f64 {
        x.exp()
    }

    pub extern "C" fn ln(x: f64) -> f64 {
        x.ln()
    }

    pub extern "C" fn log(x: f64) -> f64 {
        x.log10()
    }

    pub extern "C" fn expm1(x: f64) -> f64 {
        x.exp_m1()
    }

    pub extern "C" fn log1p(x: f64) -> f64 {
        x.ln_1p()
    }

    pub extern "C" fn exp2(x: f64) -> f64 {
        x.exp2()
    }

    pub extern "C" fn log2(x: f64) -> f64 {
        x.log2()
    }

    pub extern "C" fn gamma(x: f64) -> f64 {
        cephes64::gamma(x)
    }

    pub extern "C" fn loggamma(x: f64) -> f64 {
        cephes64::lgam(x)
    }

    pub extern "C" fn erf(x: f64) -> f64 {
        cephes64::erf(x)
    }

    pub extern "C" fn erfc(x: f64) -> f64 {
        cephes64::erfc(x)
    }

    pub extern "C" fn si(x: f64) -> f64 {
        let (s, _) = cephes64::sici(x);
        s
    }

    pub extern "C" fn ci(x: f64) -> f64 {
        let (_, c) = cephes64::sici(x);
        c
    }

    pub extern "C" fn shi(x: f64) -> f64 {
        let (s, _) = cephes64::shichi(x);
        s
    }

    pub extern "C" fn chi(x: f64) -> f64 {
        let (_, c) = cephes64::shichi(x);
        c
    }

    pub extern "C" fn random(x: f64) -> f64 {
        rand::random::<f64>() * x
    }

    pub extern "C" fn arg(_x: f64) -> f64 {
        0.0
    }

    pub extern "C" fn stable_abs(x: f64) -> f64 {
        x.abs()
    }

    pub extern "C" fn stable_divide(x: f64, y: f64) -> f64 {
        x / y
    }

    pub extern "C" fn stable_root(x: f64) -> f64 {
        x.sqrt()
    }

    /************** Complex Functions ***************/

    pub extern "C" fn cplx_stable_abs(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr.hypot(xi), 0.0);
    }

    pub extern "C" fn cplx_stable_divide(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = divide_complex(Complex::new(xr, xi), *z);
    }

    pub extern "C" fn cplx_sinc(xr: f64, xi: f64, z: &mut Complex<f64>) {
        let x = Complex::new(xr, xi);
        if x == Complex::ZERO {
            *z = Complex::ONE;
        } else {
            *z = x.sin() / x;
        }
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_sin(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { csin(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_cos(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { ccos(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_tan(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { ctan(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_csc(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(unsafe { csin(Complex::new(xr, xi)) });
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_sec(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(unsafe { ccos(Complex::new(xr, xi)) });
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_cot(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(unsafe { ctan(Complex::new(xr, xi)) });
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_sinh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { csinh(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_cosh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { ccosh(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_tanh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { ctanh(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_csch(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(unsafe { csinh(Complex::new(xr, xi)) });
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_sech(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(unsafe { ccosh(Complex::new(xr, xi)) });
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_coth(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(unsafe { ctanh(Complex::new(xr, xi)) });
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_asin(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { casin(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_acos(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { cacos(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_atan(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { catan(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_asinh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { casinh(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_acosh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { cacosh(Complex::new(xr, xi)) };
    }

    #[cfg(all(target_family = "unix", feature = "libm"))]
    pub extern "C" fn cplx_atanh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = unsafe { catanh(Complex::new(xr, xi)) };
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_sin(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).sin();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_cos(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).cos();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_tan(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = tan_c(Complex::new(xr, xi));
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_csc(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(Complex::new(xr, xi).sin());
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_sec(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(Complex::new(xr, xi).cos());
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_cot(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(tan_c(Complex::new(xr, xi)));
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_sinh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).sinh();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_cosh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).cosh();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_tanh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = tanh_c(Complex::new(xr, xi));
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_csch(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(Complex::new(xr, xi).sinh());
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_sech(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(Complex::new(xr, xi).cosh());
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_coth(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = recip(tanh_c(Complex::new(xr, xi)));
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_asin(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).asin();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_acos(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).acos();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_atan(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).atan();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_asinh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).asinh();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_acosh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).acosh();
    }

    #[cfg(any(not(target_family = "unix"), not(feature = "libm")))]
    pub extern "C" fn cplx_atanh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).atanh();
    }

    pub extern "C" fn cplx_root(xr: f64, xi: f64, z: &mut Complex<f64>) {
        if xi == 0.0 || !xr.is_finite() || !xi.is_finite() {
            *z = Complex::new(xr, xi).sqrt();
            return;
        }
        // An even binary scale keeps both the norm and the half-sum normal.
        // Recover the smaller component from the original imaginary part:
        // normalizing both output components would erase finite tiny companions.
        let (_, exponent) = finite_parts(xr.abs().max(xi.abs()));
        let half_exponent = exponent.div_euclid(2);
        let re = scale_binary(xr, -2 * half_exponent);
        let im = scale_binary(xi, -2 * half_exponent);
        let larger = scale_binary(((re.hypot(im) + re.abs()) * 0.5).sqrt(), half_exponent);
        *z = if xr >= 0.0 {
            Complex::new(larger, xi / (2.0 * larger))
        } else {
            Complex::new(xi.abs() / (2.0 * larger), larger.copysign(xi))
        };
    }

    pub extern "C" fn cplx_cbrt(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).cbrt();
    }

    pub extern "C" fn cplx_exp(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).exp();
    }

    pub extern "C" fn cplx_ln(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).ln();
    }

    pub extern "C" fn cplx_log(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).ln();
    }

    pub extern "C" fn cplx_power(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).powc(*z);
    }

    pub extern "C" fn cplx_random(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(rand::random::<f64>(), rand::random::<f64>()) * Complex::new(xr, xi);
    }

    pub extern "C" fn cplx_arg(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(Complex::new(xr, xi).arg(), 0.0);
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;
    use num_rational::BigRational;
    use num_traits::ToPrimitive;

    #[test]
    fn complex_division_matches_exact_rational_components_across_finite_range() {
        let minimum = f64::from_bits(1);
        let mut inputs = vec![
            (Complex::new(1e308, minimum), Complex::new(1., 0.)),
            (Complex::new(minimum, 1e308), Complex::new(0., 1.)),
            (Complex::new(1e308, minimum), Complex::new(1., minimum)),
            (Complex::new(1e308, 1e308), Complex::new(1e-320, 1e-320)),
            (Complex::new(minimum, minimum), Complex::new(1e308, 1e308)),
            (Complex::new(1e-200, 1e200), Complex::new(1e200, 1e-200)),
        ];
        for scale in [minimum, 1e-320, 1e-200, 0.5, 1e200, 1e307] {
            inputs.extend([
                (
                    Complex::new(3. * scale, 4. * scale),
                    Complex::new(scale, -2. * scale),
                ),
                (Complex::new(scale, -scale), Complex::new(scale, scale)),
                (Complex::new(scale, scale), Complex::new(scale, 0.)),
            ]);
        }
        for (x, y) in inputs {
            let exact = |value| BigRational::from_float(value).unwrap();
            let (a, b, c, d) = (exact(x.re), exact(x.im), exact(y.re), exact(y.im));
            let denominator = &c * &c + &d * &d;
            let reference = [
                ((&a * &c + &b * &d) / &denominator).to_f64().unwrap(),
                ((&b * &c - &a * &d) / &denominator).to_f64().unwrap(),
            ];
            let result = divide_complex(x, y);
            for (actual, expected) in [result.re, result.im].into_iter().zip(reference) {
                if expected.is_infinite() || expected == 0.0 {
                    assert_eq!(actual, expected, "{x}/{y}");
                } else {
                    assert!(
                        (actual - expected).abs()
                            <= 8. * f64::EPSILON * expected.abs() + 2. * minimum,
                        "{x}/{y}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn complex_range_callbacks_preserve_axes_and_magnitude() {
        for (x, y) in [
            (Complex::new(-0., 1e-320), Complex::new(-2., 0.)),
            (Complex::new(1e308, -0.), Complex::new(0., -2.)),
        ] {
            let expected = if y.im == 0. {
                Complex::new(x.re / y.re, x.im / y.re)
            } else {
                Complex::new(x.im / y.im, -x.re / y.im)
            };
            let mut actual = y;
            VirtualTable::cplx_stable_divide(x.re, x.im, &mut actual);
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
        for (re, im) in [
            (0., -0.),
            (1e-320, 0.),
            (3e-200, 4e-200),
            (3e200, 4e200),
            (1e308, 1e308),
            (f64::INFINITY, 1.),
            (f64::NAN, 1.),
        ] {
            let mut actual = Complex::default();
            VirtualTable::cplx_stable_abs(re, im, &mut actual);
            let expected = re.hypot(im);
            assert!(actual.re == expected || (actual.re.is_nan() && expected.is_nan()));
            assert_eq!(actual.im.to_bits(), 0.0f64.to_bits());
        }
    }
}

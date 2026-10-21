//! Win64 makes the low 128 bits of xmm6-xmm15 (and rsi, rdi, rbx, rbp, r12-r15) non-volatile.
//! The AMD kernels kept only the low 64 bits of xmm6-xmm15 (`scalar.rs`, `sse.rs`: 8-byte
//! `movsd` into the 8-byte spill slots `Stack[6..15]`) or saved them 16 bytes wide 8 bytes
//! apart (`complex.rs`), so a caller holding packed values in these registers lost them; the
//! Rust callers in the tests did (e.g. tests/constants.rs passed 0.0 instead of 2.0 to
//! `append_constant` after calling an earlier kernel). The kernels now save xmm6-xmm10 16
//! bytes wide in the spill slots `Stack[6..15]` and xmm11-xmm15 in the 80 bytes the
//! prologues reserve above RBP; a kernel frame without that reservation writes the caller's
//! stack instead (the Windows fast kernels did).
//!
//! Every kernel here (scalar, SIMD and fast; AVX, SSE, fast-complex and complexified) is
//! called through a trampoline that loads sentinels into the non-volatile registers and into
//! 128 bytes of the caller's frame above the 32-byte home area, and checks them after the
//! call. The models keep many values live, so the kernels use xmm6-xmm15.
#![cfg(all(target_os = "windows", target_arch = "x86_64"))]
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use std::arch::asm;
use symjit::{Compiled, Compiler, CompilerType, Composer, Config, Expr, Slot, Translator};

const GP_NAMES: [&str; 6] = ["rsi", "rdi", "r12", "r13", "r14", "r15"];

fn sentinel(k: usize) -> [u64; 2] {
    let k = k as u64;
    [
        0x5a5a_0000_0000_00cc | (k << 8),
        0xa5a5_0000_0000_00bb | (k << 8),
    ]
}

/// The 8-byte words of the caller's frame above the home area that `call_checked` watches.
const FRAME_WORDS: usize = 16;
const FRAME_SENTINEL: u64 = 0xc0de_cafe_f00d_0000;

/// Calls `f` with `gp` in rcx, rdx, r8, r9 and `x0`, `x1` in xmm0, xmm1 (so it serves the
/// indirect kernels and the fast ones), sentinels in the non-volatile registers and in the
/// `FRAME_WORDS` words above the home area. Returns xmm0 and what the kernel changed.
fn call_checked(f: usize, gp: [usize; 4], x0: f64, x1: f64) -> (f64, Vec<String>) {
    let mut xmm: [[u64; 2]; 10] = std::array::from_fn(|k| sentinel(k + 6));
    let mut frame = [0u64; FRAME_WORDS];
    // regs[0..6] = sentinels loaded before the call, regs[6..12] = values after it,
    // regs[12] = the xmm buffer pointer across the call, regs[13] = the frame buffer pointer
    let mut regs: [u64; 14] = std::array::from_fn(|k| 0x1111_1111_0000_00a0 * (k as u64 % 6 + 1));
    regs[13] = frame.as_mut_ptr() as u64;
    let ret: f64;

    unsafe {
        asm!(
            "push rbp",
            "push rbx",
            "mov rbx, rax",
            "mov rbp, rsp",
            "and rsp, -16",
            // [rsp, rsp + 32) is the home area, [rsp + 32, rsp + 160) belongs to the caller
            "sub rsp, 32 + 8 * {words}",
            "mov rsi, [rbx]",
            "mov rdi, [rbx + 8]",
            "mov r12, [rbx + 16]",
            "mov r13, [rbx + 24]",
            "mov r14, [rbx + 32]",
            "mov r15, [rbx + 40]",
            "movdqu xmm6, [r11]",
            "movdqu xmm7, [r11 + 16]",
            "movdqu xmm8, [r11 + 32]",
            "movdqu xmm9, [r11 + 48]",
            "movdqu xmm10, [r11 + 64]",
            "movdqu xmm11, [r11 + 80]",
            "movdqu xmm12, [r11 + 96]",
            "movdqu xmm13, [r11 + 112]",
            "movdqu xmm14, [r11 + 128]",
            "movdqu xmm15, [r11 + 144]",
            "mov [rbx + 96], r11",
            "xor r11d, r11d",
            "2:",
            "mov rax, {frame_sentinel}",
            "or rax, r11",
            "mov [rsp + 32 + 8 * r11], rax",
            "inc r11",
            "cmp r11, {words}",
            "jne 2b",
            "call r10",
            "mov r11, [rbx + 104]",
            "xor ecx, ecx",
            "3:",
            "mov rax, [rsp + 32 + 8 * rcx]",
            "mov [r11 + 8 * rcx], rax",
            "inc rcx",
            "cmp rcx, {words}",
            "jne 3b",
            "mov r11, [rbx + 96]",
            "movdqu [r11], xmm6",
            "movdqu [r11 + 16], xmm7",
            "movdqu [r11 + 32], xmm8",
            "movdqu [r11 + 48], xmm9",
            "movdqu [r11 + 64], xmm10",
            "movdqu [r11 + 80], xmm11",
            "movdqu [r11 + 96], xmm12",
            "movdqu [r11 + 112], xmm13",
            "movdqu [r11 + 128], xmm14",
            "movdqu [r11 + 144], xmm15",
            "mov [rbx + 48], rsi",
            "mov [rbx + 56], rdi",
            "mov [rbx + 64], r12",
            "mov [rbx + 72], r13",
            "mov [rbx + 80], r14",
            "mov [rbx + 88], r15",
            "mov rsp, rbp",
            "pop rbx",
            "pop rbp",
            words = const FRAME_WORDS,
            frame_sentinel = const FRAME_SENTINEL,
            in("r10") f,
            in("r11") xmm.as_mut_ptr(),
            in("rax") regs.as_mut_ptr(),
            in("rcx") gp[0], in("rdx") gp[1], in("r8") gp[2], in("r9") gp[3],
            inout("xmm0") x0 => ret,
            in("xmm1") x1,
            out("rsi") _, out("rdi") _, out("r12") _, out("r13") _, out("r14") _, out("r15") _,
            out("xmm6") _, out("xmm7") _, out("xmm8") _, out("xmm9") _, out("xmm10") _,
            out("xmm11") _, out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
            clobber_abi("C"),
        );
    }

    let mut clobbered = Vec::new();
    for (k, x) in xmm.iter().enumerate() {
        if *x != sentinel(k + 6) {
            clobbered.push(format!("xmm{} = {:016x}:{:016x}", k + 6, x[1], x[0]));
        }
    }
    for k in 0..6 {
        if regs[k] != regs[k + 6] {
            clobbered.push(format!("{} = {:016x}", GP_NAMES[k], regs[k + 6]));
        }
    }
    for (k, word) in frame.iter().enumerate() {
        if *word != FRAME_SENTINEL | k as u64 {
            clobbered.push(format!("caller frame [rsp + {}] = {word:016x}", 32 + 8 * k));
        }
    }
    (ret, clobbered)
}

fn config(ty: CompilerType) -> Config {
    let mut config = base_config();
    config.ty = ty;
    config
}

/// sum t_i * t_(i + 1) with t_i = (i + 1) x + (i + 2) y: the t_i are live at the same time
fn pressure(x: &Expr, y: &Expr) -> Expr {
    let t: Vec<Expr> = (0..12)
        .map(|i| &(x * &Expr::from((i + 1) as f64)) + &(y * &Expr::from((i + 2) as f64)))
        .collect();
    let mut sum = Expr::from(0.0);
    for i in 0..t.len() {
        sum = &sum + &(&t[i] * &t[(i + 1) % t.len()]);
    }
    sum
}

fn pressure_value(x: f64, y: f64) -> f64 {
    let t: Vec<f64> = (0..12)
        .map(|i| (i + 1) as f64 * x + (i + 2) as f64 * y)
        .collect();
    (0..12).map(|i| t[i] * t[(i + 1) % 12]).sum()
}

const STATES: [f64; 2] = [0.75, -1.25];

/// The kernels may sum the terms in another order.
#[track_caller]
fn assert_close(value: f64, expected: f64, tag: &str) {
    assert!(
        (value - expected).abs() <= 1e-12 * expected.abs(),
        "{tag}: {value} != {expected}"
    );
}

/// Indirect (sympy layout), SIMD and fast kernels of a `Compiler` model.
fn check_compiler(ty: CompilerType, opt_level: u8) -> Result<usize> {
    let mut config = config(ty);
    config.set_symbolica(false);
    config.set_opt_level(opt_level);
    let x = Expr::var("x");
    let y = Expr::var("y");
    let mut app =
        Compiler::with_config(config).compile(&[x.clone(), y.clone()], &[pressure(&x, &y)])?;
    app.prepare_simd();
    let expected = pressure_value(STATES[0], STATES[1]);
    let tag = format!("{ty:?} Compiler opt_level={opt_level}");
    let mut checked = 0;

    // MEM = states and observables; states = null skips the gather of the vectorized form
    let mut mem = vec![0.0f64; 1 << 12];
    mem[..2].copy_from_slice(&STATES);
    let params = vec![0.0f64; 1 << 12];
    for (kind, kernel) in [("scalar", &app.compiled), ("simd", &app.compiled_simd)] {
        if let Some(kernel) = kernel {
            let f = kernel.func() as usize;
            let (_, clobbered) = call_checked(
                f,
                [mem.as_mut_ptr() as usize, 0, 0, params.as_ptr() as usize],
                0.0,
                0.0,
            );
            assert!(clobbered.is_empty(), "{tag} {kind}: {clobbered:?}");
            checked += 1;
        }
    }
    assert_close(mem[2], expected, &format!("{tag} scalar"));

    let f = app.get_fast().expect("a fast kernel") as usize;
    let (value, clobbered) = call_checked(f, [0; 4], STATES[0], STATES[1]);
    assert!(clobbered.is_empty(), "{tag} fast: {clobbered:?}");
    assert_close(value, expected, &format!("{tag} fast"));
    Ok(checked + 1)
}

/// Symbolica-layout kernels of a `Translator` model, real or complex.
fn check_translator(
    ty: CompilerType,
    complex: bool,
    fast_complex: bool,
    direct: bool,
    opt_level: u8,
) -> Result<usize> {
    let mut config = config(ty);
    config.set_complex(complex);
    config.set_fast_complex(fast_complex);
    config.set_direct(direct);
    config.set_opt_level(opt_level);
    let mut t = Translator::new(config);
    t.set_num_params(2);
    // Temp(i) = (i + 1) p0 + (i + 2) p1, Out(0) = sum Temp(i) Temp(i + 1)
    let n = 12;
    for i in 0..n {
        let a = Slot::Const(t.append_constant(Complex::new((i + 1) as f64, 0.0))?);
        let b = Slot::Const(t.append_constant(Complex::new((i + 2) as f64, 0.0))?);
        t.append_mul(&Slot::Temp(2 * n + 2 * i), &[a, Slot::Param(0)], 0)?;
        t.append_mul(&Slot::Temp(2 * n + 2 * i + 1), &[b, Slot::Param(1)], 0)?;
        t.append_add(
            &Slot::Temp(i),
            &[Slot::Temp(2 * n + 2 * i), Slot::Temp(2 * n + 2 * i + 1)],
            0,
        )?;
    }
    for i in 0..n {
        t.append_mul(
            &Slot::Temp(4 * n + i),
            &[Slot::Temp(i), Slot::Temp((i + 1) % n)],
            0,
        )?;
    }
    let terms: Vec<Slot> = (0..n).map(|i| Slot::Temp(4 * n + i)).collect();
    t.append_add(&Slot::Out(0), &terms, 0)?;
    let app = t.compile()?;

    let tag = format!(
        "{ty:?} Translator complex={complex} fast_complex={fast_complex} direct={direct} opt_level={opt_level}"
    );
    let width = if complex { 2 } else { 1 };
    let mut args = vec![0.0f64; 1 << 12];
    args[0] = STATES[0];
    args[width] = STATES[1];
    let mut checked = 0;
    for (kind, kernel) in [("scalar", &app.compiled), ("simd", &app.compiled_simd)] {
        if let Some(kernel) = kernel {
            let mut outs = vec![0.0f64; 1 << 12];
            let f = kernel.func() as usize;
            let (_, clobbered) = call_checked(
                f,
                [outs.as_mut_ptr() as usize, 0, 0, args.as_ptr() as usize],
                0.0,
                0.0,
            );
            assert!(clobbered.is_empty(), "{tag} {kind}: {clobbered:?}");
            if kind == "scalar" {
                assert_close(outs[0], pressure_value(STATES[0], STATES[1]), &tag);
            }
            checked += 1;
        }
    }
    Ok(checked)
}

fn types() -> Vec<CompilerType> {
    let mut types = vec![CompilerType::Native];
    if config(CompilerType::Native).has_avx() {
        types.push(CompilerType::AmdSSE);
    }
    types
}

#[test]
fn compiler_kernels_preserve_nonvolatile_registers() -> Result<()> {
    let mut checked = 0;
    for ty in types() {
        for opt_level in [0, 1, 2, 3] {
            checked += check_compiler(ty, opt_level)?;
        }
    }
    assert!(checked >= 12, "only {checked} kernels checked");
    Ok(())
}

#[test]
fn translator_kernels_preserve_nonvolatile_registers() -> Result<()> {
    let mut checked = 0;
    for ty in types() {
        for complex in [false, true] {
            for fast_complex in [false, true] {
                for direct in [false, true] {
                    for opt_level in [0, 2] {
                        checked += check_translator(ty, complex, fast_complex, direct, opt_level)?;
                    }
                }
            }
        }
    }
    assert!(checked >= 16, "only {checked} kernels checked");
    Ok(())
}

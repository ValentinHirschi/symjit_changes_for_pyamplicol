//! Saved applications of sjb version 3, written by symjit 2.22 to 2.26 (e.g. the kernels of
//! pyAmpliCol, which pins 2.26.4). Their `Config` holds only the magic number and the option word;
//! version 4 (2.27 on) added the stack limit and the argument count. Until 2026-10-21 the loader
//! read a stack limit for version 3 as well, one word too many, and every version-3 file failed
//! with `invalid magic number (Program)`. Checked against 2.26.4 itself on 2026-10-21: the 170
//! applications of pyAmpliCol 1.0.1's prepared Standard Model pack (plain and plane kernels) give
//! the bits 2.26.4 gives after the fix.
mod common;
use common::base_config;

use symjit::{Application, Compiler, Defuns, Expr, Storage};

/// Written by symjit 2.26.4 (`Compiler::new().compile_params(&[], &[x * y + sin(x), x - y / 3],
/// &[x, y])`, `prepare_simd`, `save`): parameters x, y; outputs x*y + sin(x), x - y/3.
const SJB_2_26_4: [u8; 229] = [
    0xe9, 0x08, 0x0d, 0x41, 0x95, 0x87, 0x56, 0x40, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x5d, 0x4d, 0x8e, 0xfb, 0xae, 0x44, 0xb2, 0xc2, 0x70, 0x2e, 0x42, 0xc7, 0x77, 0x3c, 0x2c, 0x80,
    0x4d, 0x92, 0xa0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xbc, 0x27, 0xe6, 0xab, 0x25, 0x1e, 0xf2, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x31, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x03, 0x73, 0x69, 0x6e, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9e, 0x0c, 0xb0, 0x23,
    0x63, 0x9b, 0x6a, 0x87, 0x31, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xc2, 0x81, 0x00,
    0x0b, 0x01, 0x03, 0x73, 0x69, 0x6e, 0x04, 0xc0, 0xc1, 0x30, 0x03, 0x40, 0x81, 0x00, 0x26, 0x41,
    0x40, 0x81, 0x01, 0x24, 0x41, 0x41, 0xc1, 0x30, 0x04, 0x41, 0x41, 0x00, 0x03, 0x42, 0x81, 0x01,
    0x2b, 0x43, 0x42, 0x01, 0x00, 0x81, 0x43, 0x40, 0x43, 0x04, 0x43, 0x41, 0x01, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00,
];

/// The points (x, y), and what 2.26.4 computes from the file on x86-64 (`evaluate_matrix`).
const POINTS: [f64; 6] = [0.5, 2.0, -1.25, 0.75, 3.0, -4.0];
const BITS_2_26_4: [u64; 6] = [
    0x3ff7abba1d12c17c, // 1.479425538604203
    0xbfc5555555555554, // -0.16666666666666663
    0xbffe2f0a7f08a0c6, // -1.8864846193555862
    0xbff8000000000000, // -1.5
    0xc027b7bf1e4924ab, // -11.858879991940134
    0x4011555555555555, // 4.333333333333333
];

fn load(bytes: &[u8]) -> anyhow::Result<Application> {
    let mut config = base_config();
    config.set_defuns(Defuns::new());
    Application::load(&mut &bytes[..], &config)
}

fn evaluate(app: Application) -> anyhow::Result<[f64; 6]> {
    let applet = app.seal()?;
    let mut outs = [0.0; 6];
    applet.evaluate_matrix(&POINTS, &mut outs, 3);
    Ok(outs)
}

#[test]
fn version_3_files_load() -> anyhow::Result<()> {
    let app = load(&SJB_2_26_4)?;
    assert_eq!((app.count_states, app.count_params, app.count_obs), (0, 2, 2));
    // version 3 saves neither: the defaults
    assert_eq!(app.config.stack_limit(), 1 << 20);
    assert_eq!(app.config.num_args(), base_config().num_args());

    let outs = evaluate(app)?;
    for (k, (x, y)) in POINTS.chunks(2).map(|p| (p[0], p[1])).enumerate() {
        let want = [x * y + x.sin(), x - y / 3.0];
        for j in 0..2 {
            let got = outs[2 * k + j];
            assert!(
                (got - want[j]).abs() <= 1e-15 * want[j].abs().max(1.0),
                "point {k}, output {j}: {got} != {}",
                want[j]
            );
            // the machine code compiled from the file is the one 2.26.4 compiles
            #[cfg(target_arch = "x86_64")]
            assert_eq!(got.to_bits(), BITS_2_26_4[2 * k + j], "point {k}, output {j}");
        }
    }
    Ok(())
}

#[test]
fn version_3_files_are_saved_as_version_4() -> anyhow::Result<()> {
    let mut bytes = Vec::new();
    load(&SJB_2_26_4)?.save(&mut bytes)?;
    assert_eq!(bytes[8..16], 4usize.to_le_bytes());
    assert_eq!(evaluate(load(&bytes)?)?, evaluate(load(&SJB_2_26_4)?)?);
    Ok(())
}

#[test]
fn version_4_keeps_the_stack_limit() -> anyhow::Result<()> {
    let mut config = base_config();
    config.set_symbolica(false);
    config.set_stack_limit(1 << 21);
    let (x, y) = (Expr::var("x"), Expr::var("y"));
    let app = Compiler::with_config(config).compile_params(
        &[],
        &[x.mul(&y).add(&x.sin()), x.sub(&y.div(&Expr::from(3.0)))],
        &[x.clone(), y.clone()],
    )?;
    let mut bytes = Vec::new();
    app.save(&mut bytes)?;
    let app = load(&bytes)?;
    assert_eq!(app.config.stack_limit(), 1 << 21);
    let outs = evaluate(app)?;
    assert!((outs[0] - (0.5 * 2.0 + 0.5f64.sin())).abs() < 1e-15);
    Ok(())
}

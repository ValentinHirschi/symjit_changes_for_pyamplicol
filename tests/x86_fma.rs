//! The AVX generators also emit FMA3 instructions (complex products, the `fastmath`
//! fusions), so the AVX code is chosen only on CPUs with both AVX and FMA3. Other x86-64
//! CPUs, e.g. Intel Sandy and Ivy Bridge (AVX without FMA3), get the SSE code; before this
//! check they got the AVX code, and complex products like the one below raised SIGILL
//! (reproducible with `qemu-x86_64 -cpu SandyBridge`).
#![cfg(target_arch = "x86_64")]
mod common;
use common::base_config;

use symjit::{Complex, Composer, Slot, Translator};

#[test]
fn avx_code_only_with_fma3() -> anyhow::Result<()> {
    let mut config = base_config();
    if config.has_avx() {
        assert!(is_x86_feature_detected!("fma"));
    }

    config.set_complex(true);
    config.set_fastmath(true);
    let mut translator = Translator::new(config);
    translator.set_num_params(2);
    translator.append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Param(1)], 0)?;
    let applet = translator.compile()?.seal()?;
    let mut out = [Complex::new(0.0, 0.0)];
    applet.evaluate(&[Complex::new(1.0, 2.0), Complex::new(3.0, -1.0)], &mut out);
    assert_eq!(out[0], Complex::new(5.0, 5.0));
    Ok(())
}

//! `Config::default()` takes its options from the file named by `SYMJIT_TOML`, else from
//! `symjit.toml` in the working directory. A file that cannot be read or parsed is reported
//! and ignored: it must not panic `Config::default()`, which compilations and loads of saved
//! applications call.
//!
//! This test sets `SYMJIT_TOML` itself instead of calling `common::base_config`; it is the
//! only test in this file, so no other thread reads the environment meanwhile.
use symjit::{Application, CompilerType, Complex, Composer, Config, Slot, Storage, Translator};

/// z1 * z2, compiled with an explicit configuration and saved.
fn saved_product() -> anyhow::Result<Vec<u8>> {
    let mut config = Config::new(CompilerType::Native, 0)?;
    config.set_complex(true);
    let mut translator = Translator::new(config);
    translator.set_num_params(2);
    translator.append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Param(1)], 0)?;
    let mut bytes = Vec::new();
    translator.compile()?.save(&mut bytes)?;
    Ok(bytes)
}

fn options(config: &Config) -> (u32, usize, u32) {
    (config.opt, config.stack, config.args)
}

#[test]
fn unreadable_or_malformed_symjit_toml_is_ignored() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join(format!("symjit-config-toml-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let missing = dir.join("missing.toml");
    let malformed = dir.join("symjit.toml");
    std::fs::write(&malformed, "this is [not valid toml\n")?;
    // A path below a regular file: on Unix, checking whether it exists fails (ENOTDIR).
    let unreadable = malformed.join("symjit.toml");

    std::env::set_var("SYMJIT_TOML", &missing);
    let defaults = options(&Config::default());
    let bytes = saved_product()?;

    for path in [&malformed, &unreadable] {
        std::env::set_var("SYMJIT_TOML", path);
        assert_eq!(options(&Config::default()), defaults, "{}", path.display());

        let loader = Config::new(CompilerType::Native, 0)?;
        let applet = Application::load(&mut bytes.as_slice(), &loader)?.seal()?;
        let mut out = [Complex::new(0.0, 0.0)];
        applet.evaluate(&[Complex::new(1.0, 2.0), Complex::new(3.0, -1.0)], &mut out);
        assert_eq!(out[0], Complex::new(5.0, 5.0), "{}", path.display());
    }

    std::env::remove_var("SYMJIT_TOML");
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

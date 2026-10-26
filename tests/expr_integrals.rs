//! `Expr::si`, `ci`, `shi` and `chi` named their functions `si`, `ci`, `shi` and `chi`, which the
//! function table (`VirtualTable`, as SymPy's printer: `Si`, `Ci`, `Shi`, `Chi`) does not know:
//! compiling them failed with "op_code si is not found or is not supported".
mod common;
use common::base_config;

use anyhow::Result;
use symjit::{var, Compiler, Expr};

#[test]
fn sine_and_cosine_integrals() -> Result<()> {
    let x = var("x");
    let mut config = base_config();
    config.set_symbolica(false);
    let mut app = Compiler::with_config(config).compile(
        std::slice::from_ref(&x),
        &[
            x.si(),
            x.ci(),
            x.shi(),
            x.chi(),
            Expr::unary("Si", &x),
            Expr::unary("Ci", &x),
            Expr::unary("Shi", &x),
            Expr::unary("Chi", &x),
        ],
    )?;
    for v in [0.5, 1.0, 4.5, 20.0] {
        let y = app.call(&[v]);
        assert_eq!(y[..4], y[4..], "at {}", v);
    }
    // Cephes' (spec_math's tests) and mpmath's values
    let y = app.call(&[1.0]);
    assert_eq!(y[0], 0.9460830703671831);
    assert_eq!(y[1], 0.33740392290096816);
    assert!((y[2] - 1.0572508753757285).abs() < 1e-15, "{}", y[2]);
    assert!((y[3] - 0.8378669409802082).abs() < 1e-15, "{}", y[3]);
    Ok(())
}

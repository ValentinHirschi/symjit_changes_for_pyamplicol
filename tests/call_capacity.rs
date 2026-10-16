mod common;

use common::base_config;
use symjit::{Complex, Composer, Defuns, Slot, Translator};

fn compile_call(direct: bool, reservation: u32, arity: usize) -> anyhow::Result<symjit::Application> {
    let mut config = base_config();
    config.set_direct(direct);
    config.set_complex(false);
    config.set_num_args(reservation);
    let mut functions = Defuns::new();
    functions.add_sliced_func::<f64>(
        "sum_arguments",
        Box::new(Box::new(|arguments: &[f64]| arguments.iter().sum())),
    )?;
    config.set_defuns(functions);
    let mut translator = Translator::new(config);
    translator.set_num_params(0);
    let one = translator.append_constant(Complex::new(1., 0.))?;
    translator.append_fun(
        &Slot::Out(0),
        "sum_arguments",
        &vec![Slot::Const(one); arity],
        true,
    )?;
    translator.compile()
}

#[test]
fn call_arguments_require_reserved_storage_in_both_translators() {
    for direct in [false, true] {
        for (reservation, arity) in [(64, 64), (65, 65), (2048, 2048)] {
            let evaluator = compile_call(direct, reservation, arity).unwrap();
            assert_eq!(evaluator.evaluate_single::<f64>(&[]), arity as f64);
        }
        for (reservation, arity) in [(64, 65), (2048, 2049), (2049, 2049)] {
            assert!(
                compile_call(direct, reservation, arity).is_err(),
                "direct={direct}, reservation={reservation}, arity={arity}"
            );
        }
    }
}

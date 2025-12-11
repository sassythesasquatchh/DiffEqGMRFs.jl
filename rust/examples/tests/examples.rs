use diffeq_gmrfs_examples::{run_burgers_example, run_darcy_example, run_shallow_water_example};

#[test]
fn burgers_example_runs() {
    let result = run_burgers_example(None).expect("burgers example should run");
    assert!(result.rmse_to_first_snapshot.is_finite());
    assert!(result.drift_norm > 0.0);
}

#[test]
fn darcy_example_runs() {
    let result = run_darcy_example(None).expect("darcy example should run");
    assert!(result.load_norm.is_finite());
    assert_eq!(result.keep_indices, 0);
}

#[test]
fn shallow_water_example_runs() {
    let result = run_shallow_water_example().expect("shallow-water example should run");
    assert!(result.state_dim > 0);
    assert!(result.constrained_dofs > 0);
}

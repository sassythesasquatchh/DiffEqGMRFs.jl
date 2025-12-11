use std::path::Path;

use diffeq_gmrfs_examples::run_burgers_example;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1);
    let path_ref = path.as_deref().map(Path::new);

    let result = run_burgers_example(path_ref)?;
    println!(
        "Burgers predictor RMSE to first snapshot: {:.6}, drift norm: {:.6}",
        result.rmse_to_first_snapshot, result.drift_norm
    );
    Ok(())
}

use std::path::Path;

use diffeq_gmrfs_examples::run_darcy_example;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1);
    let path_ref = path.as_deref().map(Path::new);
    let result = run_darcy_example(path_ref)?;
    println!(
        "Darcy load vector norm: {:.6} (kept {} boundary nodes)",
        result.load_norm, result.keep_indices
    );
    Ok(())
}

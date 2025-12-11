use diffeq_gmrfs_examples::run_shallow_water_example;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = run_shallow_water_example()?;
    println!(
        "Shallow-water joint state dimension: {}, constrained DOFs: {}",
        result.state_dim, result.constrained_dofs
    );
    Ok(())
}

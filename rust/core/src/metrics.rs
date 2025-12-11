use ndarray::{ArrayBase, Data, Dimension, Zip};
use num_traits::Float;

/// Root-mean-square error between `pred` and `soln`.
///
/// Mirrors Julia's `rmse` in `src/metrics.jl` but operates on `ndarray` views.
pub fn rmse<A, S, D>(pred: &ArrayBase<S, D>, soln: &ArrayBase<S, D>) -> A
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    assert_eq!(
        pred.shape(),
        soln.shape(),
        "arrays must have matching shapes"
    );
    let len = pred.len() as f64;
    let sum_sq: A = Zip::from(pred)
        .and(soln)
        .fold(A::zero(), |acc, &p, &s| acc + (p - s) * (p - s));
    (sum_sq / A::from(len).unwrap()).sqrt()
}

/// Maximum absolute error between `pred` and `soln`.
pub fn max_error<A, S, D>(pred: &ArrayBase<S, D>, soln: &ArrayBase<S, D>) -> A
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    assert_eq!(
        pred.shape(),
        soln.shape(),
        "arrays must have matching shapes"
    );
    Zip::from(pred)
        .and(soln)
        .fold(A::zero(), |acc, &p, &s| acc.max((p - s).abs()))
}

/// Relative error measured in the Euclidean norm.
pub fn relative_error<A, S, D>(pred: &ArrayBase<S, D>, soln: &ArrayBase<S, D>) -> A
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    assert_eq!(
        pred.shape(),
        soln.shape(),
        "arrays must have matching shapes"
    );
    let mut diff_norm_sq = A::zero();
    let mut soln_norm_sq = A::zero();

    Zip::from(pred).and(soln).for_each(|&p, &s| {
        let diff = p - s;
        diff_norm_sq = diff_norm_sq + diff * diff;
        soln_norm_sq = soln_norm_sq + s * s;
    });

    diff_norm_sq.sqrt() / soln_norm_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn computes_rmse() {
        let pred = array![1.0, 2.0, 3.0];
        let soln = array![1.0, 1.0, 2.0];
        let value = rmse(&pred, &soln);
        let expected = (2.0_f64 / 3.0_f64).sqrt();
        assert!((value - expected).abs() < 1e-12);
    }

    #[test]
    fn computes_max_error() {
        let pred = array![1.0, 2.0, 3.0];
        let soln = array![0.0, 2.5, 3.5];
        let value = max_error(&pred, &soln);
        assert!((value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn computes_relative_error() {
        let pred = array![2.0, 0.0];
        let soln = array![0.0, 1.0];
        let value = relative_error(&pred, &soln);
        assert!((value - (5f64.sqrt())).abs() < 1e-12);
    }
}

use pyo3::prelude::*;

pub mod distributions;
pub use distributions::{
    BernoulliDistribution,
    BetaDistribution,
    NegativeBinomialDistribution,
};

/// Formats the sum of two numbers as string.
//#[pyfunction]
//fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//    Ok((a + b).to_string())
//}

/// A Python module implemented in Rust.
#[pymodule]
fn mcmc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BernoulliDistribution>()?;
    m.add_class::<BetaDistribution>()?;
    m.add_class::<NegativeBinomialDistribution>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for the distributions in distributions.rs

    #[test]
    fn test_likelihood_valid_inputs() {
        let bernoulli = BernoulliDistribution::new(0.7).unwrap();
        assert!((bernoulli.pmf(1i32).unwrap() - 0.7).abs() < 1e-10);
        assert!((bernoulli.pmf(0i32).unwrap() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_likelihood_invalid_input() {
        let bernoulli = BernoulliDistribution::new(0.5).unwrap();
        let result = bernoulli.pmf(2i32);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_probability() {
        let result = BernoulliDistribution::new(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_likelihood_with_signed_integer() {
        let bernoulli = BernoulliDistribution::new(0.5).unwrap();
        assert!((bernoulli.pmf(1i32).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_likelihood_with_negative_integer() {
        let bernoulli = BernoulliDistribution::new(0.4).unwrap();
        let result = bernoulli.pmf(-1i32);
        assert!(result.is_err());
    }
}
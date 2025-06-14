
use statrs::distribution::{
    Bernoulli, 
    NegativeBinomial, 
    Beta, 
    Discrete,
    Continuous,
};
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};


#[pyclass(name = "Bernoulli", subclass)]
pub struct BernoulliDistribution {
    distribution: Bernoulli
}

#[pymethods]
impl BernoulliDistribution {

    #[new]
    pub fn new(p: f64) -> PyResult<Self> {
        let distribution = Bernoulli::new(p)
            .map_err(|e| PyValueError::new_err(format!("Creating the Bernoulli distribution failed: {}", e)))?;

        Ok(Self { distribution })
    } 

    pub fn pmf(&self, x: i32) -> PyResult<f64> {

            let x_cast: u64 = x.try_into()
                .map_err(|_e| PyValueError::new_err(format!("Observation must be 0 or 1. Got {}", x)))?;

            if x_cast > 1 {
                return Err(PyValueError::new_err(
                        format!("Observation must be 0 or 1. Got {}", x_cast)
                    )
                )
            }

            Ok(self.distribution.pmf(x_cast))
        }

    pub fn logp(&self, x: i32) -> PyResult<f64> {

            let pmf: f64 = self.pmf(x)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to evaluate PMF: {}", e)))?;

            Ok(pmf.ln())
        }
}

#[pyclass(name = "NegativeBinomial", subclass)]
pub struct NegativeBinomialDistribution {
    distribution: NegativeBinomial
}

#[pymethods]
impl NegativeBinomialDistribution {

    #[new]
    pub fn new(r: f64, p: f64) -> PyResult<Self> {
        let distribution = NegativeBinomial::new(r, p)
            .map_err(|e| PyValueError::
                new_err(format!("Creating the NegativeBinomial distribution failed: {}", e)))?;

        Ok(Self { distribution })
    } 

    pub fn pmf(&self, x: i32) -> PyResult<f64> {

            let x_cast: u64 = x.try_into()
                .map_err(|_e| PyValueError::
                    new_err(format!("Observation must be a positive integer. Got {}", x)))?;

            Ok(self.distribution.pmf(x_cast))
        }

    pub fn logp(&self, x: i32) -> PyResult<f64> {

            let pmf: f64 = self.pmf(x)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to evaluate PMF: {}", e)))?;

            Ok(pmf.ln())
        }
}

#[pyclass(name = "Beta", subclass)]
pub struct BetaDistribution {
    distribution: Beta
}

#[pymethods]
impl BetaDistribution {

    #[new]
    pub fn new(alpha: f64, beta: f64) -> PyResult<Self> {
        let distribution = Beta::new(alpha, beta)
            .map_err(|e| PyValueError::
                new_err(format!("Creating the Beta distribution failed: {}", e)))?;

        Ok(Self { distribution })
    } 

    pub fn pdf(&self, x: f64) -> PyResult<f64> {
            Ok(self.distribution.pdf(x))
        }

    pub fn logp(&self, x: f64) -> PyResult<f64> {

            let pdf: f64 = self.pdf(x)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to evaluate PDF: {}", e)))?;

            Ok(pdf.ln())
        }
}
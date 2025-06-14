
use statrs::distribution::{Bernoulli, Discrete};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;


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

    pub fn pmf(&self, x: i32) -> PyResult<f64>
        {

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

}


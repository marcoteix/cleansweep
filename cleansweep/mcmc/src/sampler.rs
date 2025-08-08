use pyo3::exceptions::PyRuntimeError;
use rayon::prelude::*;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};
use rand_distr::{Normal, Distribution};
use anyhow::Result;
use crate::mcmc_utils::SamplingResult;
pub use crate::mcmc_utils::{
    check_positive,
    Alleles,
    AltBC,
    ChainSamplingResult,
    ModelParameters,
    AcceptanceRate,
};
pub use crate::distributions::{
    BernoulliDistribution,
    BetaDistribution,
    NegativeBinomialDistribution,
    UniformDistribution,
}; 

#[pyclass(name = "MetropolisHastings", subclass)]
pub struct MetropolisHastings {
    pub n_chains: i32, // Number of sampling chains
    pub n_samples: i32, // Number of sampling rounds per chain
    pub n_burnin: i32, // Number of burnin iterations per chain
    pub n_cores: i32, // Number of cores
    pub seed: i32, // Random seed
    pub dispersion_bias: f64,
    pub alt_allele_p_proposal_sd: f64, // Initial variance used in the Normal proposal distribution
    pub dispersion_proposal_sd: f64, // Initial variance used in the Normal proposal distribution
    pub proposal_p: f64, // Initial proportion used in the Bernoulli proposal distribution
    pub target_min_acceptance_rate: f64, // Minimum desired acceptance rate
    pub target_max_acceptance_rate: f64, // Maximum desired acceptance rate
    pub block_size: i32, // Number of sites updated in the same block
    pub adaptive_step_coeff: f64, // Proportional coefficient for the adaptive step size
    pub step_size_range: f64, // Allowed range for the step size. If the initial step size is x, it is allowed to range between x/step_size_range and x*step_size_range
    rng: Vec<StdRng>,
}

#[pymethods]
impl MetropolisHastings {

    #[new]
    pub fn new(
        n_chains: i32,
        n_samples: i32,
        n_burnin: i32,
        n_cores: i32,
        dispersion_bias: f64,
        alt_allele_p_proposal_sd: f64,
        dispersion_proposal_sd: f64,
        proposal_p: f64,
        target_min_acceptance_rate: f64,
        target_max_acceptance_rate: f64,
        block_size: i32,
        adaptive_step_coeff: f64,
        step_size_range: f64,
        seed: i32,
    ) -> PyResult<Self> {
        
        check_positive(n_chains, true)?;
        check_positive(n_samples, true)?;
        check_positive(n_burnin, false)?;
        check_positive(n_cores, true)?;
        check_positive(block_size, true)?;

        // Create RNGs, one per chain
        //let rng = RwLock::new(StdRng::seed_from_u64(seed as u64));
        let rng = (0..n_chains).map(|i| StdRng::seed_from_u64((seed + i) as u64)).collect();

        Ok(
            MetropolisHastings { 
                n_chains: n_chains, 
                n_samples: n_samples, 
                n_burnin: n_burnin, 
                n_cores: n_cores ,
                dispersion_bias: dispersion_bias,
                seed: seed,
                alt_allele_p_proposal_sd: alt_allele_p_proposal_sd,
                dispersion_proposal_sd: dispersion_proposal_sd,
                proposal_p: proposal_p,
                target_min_acceptance_rate: target_min_acceptance_rate,
                target_max_acceptance_rate: target_max_acceptance_rate,
                block_size: block_size,
                rng: rng,
                adaptive_step_coeff: adaptive_step_coeff,
                step_size_range: step_size_range,
            }
        )

    }

    pub fn sample(&mut self, py: Python, alt_bc: Vec<i32>, coverage: f64) -> PyResult<Py<SamplingResult>> {

        // Sample from chains in parallel
        
        // Create input tuples with (AltBC, coverage, rng)
        let inputs: Vec<(AltBC, f64, StdRng)> = self.rng.clone().into_iter()
            .map(|rng| (AltBC { vector: alt_bc.clone() }, coverage, rng))
            .collect();

        let results = self.sample_chain_parallel(&inputs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let sampling_results = self.aggregate_results(results)?;

        Py::new(py, sampling_results)

    }
}

impl MetropolisHastings {

    fn sample_chain_parallel(&self, inputs: &Vec<(AltBC, f64, StdRng)>) -> Result<Vec<ChainSamplingResult>> {
        

        inputs.into_par_iter()
            .map(|(a, b, c)| self.sample_chain(a, *b, c.clone()))
            .collect()
    }

    fn sample_chain(&self, alt_bc: &AltBC, coverage: f64, mut rng: StdRng) -> Result<ChainSamplingResult> {

        let n_samples = alt_bc.len()?;

        // Initialize the sample arrays
        let mut dispersion_samples: Vec<f64> = vec![0.5f64; self.n_samples as usize];
        let mut alt_allele_proportion_samples: Vec<f64> = vec![0.5f64; self.n_samples as usize];
        let mut alleles_samples: Vec<u8> = vec![0; self.n_samples as usize * n_samples]; // Flat matrix n_samples x len(alt_bc)

        // Get starter values

        let allele_start_values = &mut alleles_samples[0..n_samples]; // slice of length 20

        for elem in allele_start_values.iter_mut() {
            *elem = rng.random_range(0..=1);
        }

        let mut params = ModelParameters{
            dispersion: 0.5,
            alt_allele_proportion: 0.5,
            alleles: allele_start_values.to_vec(),
            alleles_acceptance_rate: 0.0,
            dispersion_acceptance_rate: 0.0,
            alt_allele_proportion_acceptance_rate: 0.0,
            logp: -1e20,
        };

        // Keep track of the acceptance rates
        let mut alt_allele_proportion_acceptance_rate_sum: f64 = 0.0;
        let mut dispersion_acceptance_rate_sum: f64 = 0.0;
        let mut alleles_acceptance_rate_sum: f64 = 0.0;

        // Adaptive step sizes
        let mut alt_allele_p_step_size = self.alt_allele_p_proposal_sd;
        let mut dispersion_step_size = self.dispersion_proposal_sd;
        let mut alleles_step_size = self.proposal_p;

        // Total number of iterations
        let total_n_iter = self.n_burnin + self.n_samples;

        for i in 0..total_n_iter {
            
            let alleles = Alleles{vector: params.alleles.clone()};

            // Do a sampling step
            params = self.step(
                params.dispersion,
                coverage,
                params.alt_allele_proportion,
                &alleles,
                params.logp,
                &alt_bc,
                alt_allele_p_step_size,
                dispersion_step_size,
                alleles_step_size,
                &mut rng,
            )?;

            // Update step sizes based on acceptance rate
            alt_allele_proportion_acceptance_rate_sum += params.alt_allele_proportion_acceptance_rate;
            dispersion_acceptance_rate_sum += params.dispersion_acceptance_rate;
            alleles_acceptance_rate_sum += params.alleles_acceptance_rate;

            alt_allele_p_step_size = self.update_step_size(
                alt_allele_p_step_size, 
                alt_allele_proportion_acceptance_rate_sum/(i as f64), 
                0.20, 
                0.60, 
                self.alt_allele_p_proposal_sd / self.step_size_range, 
                self.alt_allele_p_proposal_sd * self.step_size_range, 
                self.adaptive_step_coeff
            );

            dispersion_step_size = self.update_step_size(
                dispersion_step_size, 
                dispersion_acceptance_rate_sum/(i as f64), 
                0.20, 
                0.60, 
                self.dispersion_proposal_sd / self.step_size_range, 
                self.dispersion_proposal_sd * self.step_size_range, 
                self.adaptive_step_coeff
            );

            alleles_step_size = self.update_step_size(
                alleles_step_size, 
                alleles_acceptance_rate_sum/(i as f64), 
                0.20, 
                0.60, 
                self.proposal_p / self.step_size_range, 
                self.proposal_p * self.step_size_range, 
                self.adaptive_step_coeff
            );

            // If we are passed the burn-in stage, keep track of sampled values
            if i >= self.n_burnin {

                let idx = (i - self.n_burnin) as usize;
                dispersion_samples[idx] = params.dispersion;
                alt_allele_proportion_samples[idx] = params.alt_allele_proportion;

                let allele_start = idx * n_samples;
                let allele_end = (idx+1) * n_samples;
                alleles_samples[allele_start..allele_end].copy_from_slice(&params.alleles);

            }
        }

        Ok(ChainSamplingResult::new(
            dispersion_samples,
            alt_allele_proportion_samples,
            alleles_samples,
            alt_allele_proportion_acceptance_rate_sum / (total_n_iter as f64),
            dispersion_acceptance_rate_sum / (total_n_iter as f64),
            alleles_acceptance_rate_sum / (total_n_iter as f64),
        )?)

    }

    // Updates step size based on acceptance rates
    fn update_step_size(
        &self,
        step_size: f64,
        acceptance_rate: f64,
        min_target: f64,
        max_target: f64,
        min_value: f64,
        max_value: f64,
        coefficient: f64,
    ) -> f64 {

        let mut new_step_size = step_size; 
        let target = (max_target - min_target)/2.0;

        if acceptance_rate < min_target {
            new_step_size -= coefficient * (target - acceptance_rate);
        } else if acceptance_rate > max_target {
            new_step_size += coefficient * (target - acceptance_rate);
        }

        if new_step_size > max_value {
            new_step_size = max_value;
        } else if new_step_size < min_value {
            new_step_size = min_value;
        }

        new_step_size
    }

    // Implements an iteration of the Metropolis-Hastings algorithm.
    fn step(
        &self,
        dispersion: f64,
        coverage: f64,
        alt_allele_proportion: f64,
        alleles: &Alleles,
        previous_logp: f64,
        alt_bc: &AltBC,
        alt_allele_p_step_size: f64,
        dispersion_step_size: f64,
        alleles_step_size: f64,
        rng: &mut StdRng,
    ) -> PyResult<ModelParameters> {

        // Keep track of acceptance rates
        let mut alt_allele_proportion_acceptance_rate = AcceptanceRate::new()?;
        let mut dispersion_acceptance_rate = AcceptanceRate::new()?;
        let mut alleles_acceptance_rate = AcceptanceRate::new()?;

        // Track the logp after each block update
        let mut logp = previous_logp;

        // Propose candidates

        // Start with the dispersion
        let proposed_dispersion = self.propose_beta(dispersion, rng, dispersion_step_size)?;
        // Transform from [0, 1] -> [0, coverage*10^1.5[ with f(x) = coverage * 10^(x*3-1.5)
        let transformed_proposed_dispersion = self.transform_dispersion(proposed_dispersion, coverage)?;
        // Check if we should accept the proposed dispersion
        let mut accept = self.is_accept(
            logp, 
            transformed_proposed_dispersion, 
            alt_allele_proportion, 
            &alleles, 
            coverage, 
            &alt_bc, 
            rng
        )?;
        // Replace if needed and update logp
        let mut new_dispersion = dispersion; 
        if accept { 
            new_dispersion = proposed_dispersion;
            dispersion_acceptance_rate.accept();
            logp = self.get_logp(
                transformed_proposed_dispersion, 
                alt_allele_proportion, 
                &alleles, 
                coverage,
                &alt_bc,
            )?;
        }
        dispersion_acceptance_rate.increment();
        // Transform the new dispersion value
        let transformed_dispersion = self.transform_dispersion(new_dispersion, coverage)?;

        // Propose a new alt allele proportion
        let proposed_alt_allele_p = self.propose_beta(alt_allele_proportion, rng, alt_allele_p_step_size)?; 

        // Check if we should accept the two updates
        accept = self.is_accept(
            logp, 
            transformed_dispersion, 
            proposed_alt_allele_p, 
            &alleles, 
            coverage, 
            &alt_bc, 
            rng,
        )?; 
        // Replace if needed and update logp
        let mut new_alt_allele_p = alt_allele_proportion;
        if accept { 
            new_alt_allele_p = proposed_alt_allele_p;
            alt_allele_proportion_acceptance_rate.accept();
            logp = self.get_logp(
                transformed_dispersion, 
                new_alt_allele_p, 
                &alleles, 
                coverage,
                &alt_bc,
            )?;
        }
        alt_allele_proportion_acceptance_rate.increment();
        
        // Propose alleles, in blocks of size self.block_size
        let mut proposed_alleles = alleles.clone()?;
        let mut new_alleles = alleles.clone()?;
        proposed_alleles.copy_from_slice(&alleles.vector, 0, proposed_alleles.len()?)?;
        new_alleles.copy_from_slice(&alleles.vector, 0, proposed_alleles.len()?)?;

        for n in 0..(proposed_alleles.len()? + self.block_size as usize - 1) / self.block_size as usize {
            let start = n * self.block_size as usize;
            let end = (start + self.block_size as usize).min(proposed_alleles.len()?);

            let original_block = proposed_alleles.vector[start..end].to_vec();

            // Flip allele values with p = self.proposal_p
            for x in &mut proposed_alleles.vector[start..end] {
                let a: f64 = rng.random();
                if a < alleles_step_size {
                    *x = 1u8 - *x;
                }
            }

            // Check if we should accept
            accept = self.is_accept(logp, transformed_dispersion, new_alt_allele_p, 
                &proposed_alleles, coverage, &alt_bc, rng)?;

            // If the new proposal would lead to all 0s or all 1s, reject
            let sum: u64 = proposed_alleles.vector.iter().map(|&e| e as u64).sum();
        
            if sum == 0u64 || sum == (proposed_alleles.len()? as u64) {
                accept = false;
            }

            let block = &mut proposed_alleles.vector[start..end];

            // Replace if needed and update logp
            if accept { 
                new_alleles.copy_from_slice(block, start, end)?;
                alleles_acceptance_rate.accept();
                logp = self.get_logp(transformed_dispersion, new_alt_allele_p, &new_alleles, coverage, &alt_bc)?;        
            
            } else {
                // Reset changes to proposed_alleles
                block.copy_from_slice(&original_block);
            }
            alleles_acceptance_rate.increment();
            
        }

        Ok(
            ModelParameters::new(
                new_dispersion, 
                new_alt_allele_p, 
                new_alleles.vector, 
                alt_allele_proportion_acceptance_rate.get_acceptance_rate(),
                dispersion_acceptance_rate.get_acceptance_rate(),
                alleles_acceptance_rate.get_acceptance_rate(), 
                logp
            )?
        )
    }

    fn transform_dispersion(&self, dispersion: f64, coverage: f64) -> PyResult<f64> {
        Ok(coverage * 10f64.powf(dispersion*3.0 - 1.5))
    }

    fn inverse_transform_dispersion(&self, dispersion: f64, coverage: f64) -> PyResult<f64> {
       Ok(((dispersion/coverage).log10() + 1.5)/3.0)
    }

    fn propose_beta(&self, x: f64, rng: &mut StdRng, step_size: f64) -> PyResult<f64> {

        let normal = Normal::new(0.0, step_size).unwrap();
        let proposal = x + normal.sample(rng);

        Ok(proposal.min(0.999).max(1e-10))

    }

    fn is_accept(
        &self,
        previous_logp: f64,
        dispersion: f64,
        alt_allele_proportion: f64,
        alleles: &Alleles,
        coverage: f64,
        alt_bc: &AltBC,
        rng: &mut StdRng,
    ) -> PyResult<bool> {

        let new_logp = self.get_logp(dispersion, alt_allele_proportion, 
            alleles, coverage, alt_bc)?;

        let p_ratio = (new_logp - previous_logp).exp().min(1.0);

        // Draw a random number between 0 and 1. Accept if it is lower than the likelihood ratio
        let draw: f64 = rng.random();

        let accept = draw < p_ratio;

        Ok(accept)

    }

    fn get_logp(
        &self,
        dispersion: f64,
        alt_allele_proportion: f64,
        alleles: &Alleles,
        coverage: f64,
        alt_bc: &AltBC,
    ) -> PyResult<f64> {

        let max_alt_bc = *alt_bc.vector.iter().max().unwrap();

        let negative_binomial = NegativeBinomialDistribution::new(
            dispersion, dispersion/(dispersion+coverage))?;
        let uniform = UniformDistribution::new(0, max_alt_bc + 2)?;
        let bernoulli = BernoulliDistribution::new(alt_allele_proportion)?;
        let dispersion_prior = BetaDistribution::new(self.dispersion_bias, self.dispersion_bias)?;
        let alt_allele_p_prior = BetaDistribution::new(1.0, 1.0)?;

        let mut logp = 0.0;

        for (site, bc) in alleles.vector.iter().zip(alt_bc.vector.iter()) {

            let nb = (negative_binomial.logp(*bc)?).max(-1e20);
            
            logp += (*site as f64) * nb + (1.0-*site as f64) * 
                uniform.logp(*bc)? + bernoulli.logp(*site as i32)?;
        }

        // Add priors
        logp += dispersion_prior.logp(self.inverse_transform_dispersion(dispersion, coverage)?)? + 
            alt_allele_p_prior.logp(alt_allele_proportion)?;

        Ok(logp)

    }

    fn aggregate_results(&self, results: Vec<ChainSamplingResult>) -> PyResult<SamplingResult> {

        let aggregate_results = Python::with_gil(|py| {
            let py_results: PyResult<Vec<Py<ChainSamplingResult>>> = results
                .into_iter()
                .map(|r| Py::new(py, r))
                .collect();

            py_results
        })?;

        Ok(SamplingResult{results: aggregate_results})
    }
}

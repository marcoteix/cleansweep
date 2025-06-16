use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};
use rand_distr::{Normal, Distribution};
pub use crate::mcmc_utils::{
    check_positive,
    Alleles,
    AltBC,
};
pub use crate::distributions::{
    BernoulliDistribution,
    BetaDistribution,
    NegativeBinomialDistribution,
    UniformDistribution,
}; 

#[pyclass(name = "MetrpolisHastings", subclass)]
pub struct MetropolisHastings {
    n_chains: i32, // Number of sampling chains
    n_samples: i32, // Number of sampling rounds per chain
    n_burnin: i32, // Number of burnin iterations per chain
    n_cores: i32, // Number of cores
    seed: i32, // Random seed
    dispersion_bias: f64,
    proposal_sd: f64, // Variance used in the Normal proposal distribution
    proposal_p: f64, // Proportion used in the Bernoulli proposal distribution
    block_size: i32, // Number of sites updated in the same block
    rng: StdRng,
}

#[pyclass(name = "SamplingResult", subclass)]
pub struct SamplingResult {
    posterior: Vec<Chain>,
    mean_dispersion: f64,
    mean_lt_allele_proportion: f64
}

#[pyclass(name = "Chain", subclass)]
pub struct Chain {
    dispersion: f64,
    alleles: Vec<Vec<i8>>,
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
        proposal_sd: f64,
        proposal_p: f64,
        block_size: i32,
        seed: i32,
    ) -> PyResult<Self> {
        
        check_positive(n_chains, true)?;
        check_positive(n_samples, true)?;
        check_positive(n_burnin, false)?;
        check_positive(n_cores, true)?;
        check_positive(block_size, true)?;

        // Create an RNG
        let rng = StdRng::seed_from_u64(seed as u64);

        Ok(
            MetropolisHastings { 
                n_chains: n_chains, 
                n_samples: n_samples, 
                n_burnin: n_burnin, 
                n_cores: n_cores ,
                dispersion_bias: dispersion_bias,
                seed: seed,
                proposal_sd: proposal_sd,
                proposal_p: proposal_p,
                block_size: block_size,
                rng: rng,
            }
        )

    }

    fn sample_chain(&self, alt_bc: Vec<i32>, coverage: f64) -> PyResult<SamplingResult> {

        // Initialize the sample arrays
        let mut dispersion_samples: Vec<f64> = vec![0.5f64; self.n_samples as usize];
        let mut alt_allele_proportion_samples: Vec<f64> = vec![0.5f64; self.n_samples as usize];
        let mut alleles_samples: Vec<u8> = vec![0; self.n_samples as usize * alt_bc.len()];

        // Get starter values

        let allele_start_values = &mut alleles_samples[0..alt_bc.len()]; // slice of length 20

        for elem in allele_start_values.iter_mut() {
            *elem = self.rng.random_range(0..=1);
        }

        for i in 0..self.n_burnin {
            self.step()
        }

    }

    // Implements an iteration of the Metropolis-Hastings algorithm.
    fn step(
        &mut self,
        dispersion: f64,
        coverage: f64,
        alt_allele_proportion: f64,
        alleles: PyRef<Alleles>,
        previous_logp: f64,
        alt_bc: PyRef<AltBC>,
    ) -> PyResult<()> {

        // Kep track of the number of accepted proposals
        let mut n_accepted = 0;

        // Propose candidates

        // Start with the dispersion
        let proposed_dispersion = self.propose_beta(dispersion)?;
        // Transform from [0, 1] -> [0, coverage*10^1.5[ with f(x) = coverage * 10^(x*3-1.5)
        let mut transformed_dispersion = self.transform_dispersion(dispersion, coverage)?;
        // Check if we should accept the new value
        let mut accept = self.is_accept(previous_logp, transformed_dispersion, 
            alt_allele_proportion, alleles, coverage, alt_bc)?;
        // Replace value if needed
        let mut new_dispersion = dispersion; 
        if accept {
            new_dispersion = proposed_dispersion;
            n_accepted += 1;
        }
        // Transform the new dispersion value
        transformed_dispersion = self.transform_dispersion(new_dispersion, coverage)?;

        // Propose a new alt allele proportion
        let proposed_alt_allele_p = self.propose_beta(alt_allele_proportion)?;
        // Check if we should accept
        accept = self.is_accept(previous_logp, transformed_dispersion, 
            proposed_alt_allele_p, alleles, coverage, alt_bc)?;
        // Replace if needed
        let mut new_alt_allele_p = alt_allele_proportion;
        if accept { 
            new_alt_allele_p = proposed_alt_allele_p;
            n_accepted += 1;
        }

        // Propose alleles, in blocks of size self.block_size
        let mut proposed_alleles = vec![0u8; alleles.vector.len()];
        proposed_alleles.copy_from_slice(&alleles.vector);

        for block in proposed_alleles.chunks(self.block_size as usize) {

        }

        Ok(())

    }

    fn transform_dispersion(&self, dispersion: f64, coverage: f64) -> PyResult<f64> {
        Ok(coverage * 10f64.powf(dispersion*3.0 - 1.5))
    }

    fn inverse_transform_dispersion(&self, dispersion: f64, coverage: f64) -> PyResult<f64> {
       Ok(((dispersion/coverage).log10() + 1.5)/3.0)
    }

    fn propose_beta(&mut self, x: f64) -> PyResult<f64> {

        let normal = Normal::new(0.0, self.proposal_sd).unwrap();
        let proposal = x + normal.sample(&mut self.rng);

        Ok(proposal.min(0.999).max(1e-10))

    }

    fn is_accept(
        &mut self,
        previous_logp: f64,
        dispersion: f64,
        alt_allele_proportion: f64,
        alleles: PyRef<Alleles>,
        coverage: f64,
        alt_bc: PyRef<AltBC>,
    ) -> PyResult<bool> {

        let new_logp = self.get_logp(dispersion, alt_allele_proportion, 
            alleles.vector.clone(), coverage, alt_bc.vector.clone())?;

        let p_ratio = new_logp - previous_logp;

        // Draw a random number between 0 and 1. Accept if it is lower than the likelihood ratio
        let draw: f64 = self.rng.random();

        let accept = draw.exp() > p_ratio;

        Ok(accept)

    }

    pub fn get_logp(
        &self,
        dispersion: f64,
        alt_allele_proportion: f64,
        alleles: Vec<u8>,
        coverage: f64,
        alt_bc: Vec<i32>,
    ) -> PyResult<f64> {

        let max_alt_bc = *alt_bc.iter().max().unwrap();

        let negative_binomial = NegativeBinomialDistribution::new(
            dispersion, dispersion/(dispersion+coverage))?;
        let uniform = UniformDistribution::new(0, max_alt_bc + 2)?;
        let bernoulli = BernoulliDistribution::new(alt_allele_proportion)?;
        let dispersion_prior = BetaDistribution::new(self.dispersion_bias, self.dispersion_bias)?;
        let alt_allele_p_prior = BetaDistribution::new(1.0, 1.0)?;

        let mut logp = 0.0;

        for (site, bc) in alleles.iter().zip(alt_bc.iter()) {
            logp += (*site as f64) * negative_binomial.logp(*bc)? + (1.0-*site as f64) * 
                uniform.logp(*bc)? + bernoulli.logp(*site as i32)?;
        }

        // Add priors
        logp += dispersion_prior.logp(self.inverse_transform_dispersion(dispersion, coverage)?)? + 
            alt_allele_p_prior.logp(alt_allele_proportion)?;

        Ok(logp)

    }
}
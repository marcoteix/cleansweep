#%%
from typing import Union
import numpy as np
import pymc as pm
from pymc.exceptions import SamplingError
from dataclasses import dataclass
import pandas as pd
import pytensor.tensor as pt
import logging

@dataclass
class BaseCountFilter:

    chains: int = 5
    draws: int = 100000
    burn_in: int = 1000
    random_state: int = 23
    power: float = 0.95
    threads: Union[int, None] = 5
    engine: str = "pymc"
    overdispersion_bias: int = 1
    max_overdispersion: float = 0.7

    def __post_init__(self):

        # Convert power to quantiles
        self.__quantiles = (
            (1 - self.power)/2,
            (1 + self.power)/2
        )

    def fit_mcmc(
        self, 
        vcf: pd.DataFrame, 
        query_coverage_estimate: float,
        downsampling: Union[int, float] = 1.0,
    ) -> pd.Series:
        
        logging.debug(
            "Estimating the overdispersion for the depth of coverage of the query strain."
        )

        logging.info(
            f"Downsampling the input VCF to {downsampling} entries..."
        )

        # Downsample the VCF file
        vcf_fit = self.__downsample_vcf(
            vcf, 
            n_lines = downsampling
        )

        coordinates = self.__get_coordinates()

        with pm.Model(coords=coordinates) as model:

            model.add_coord(
                "sites", 
                vcf_fit.index.to_list(), 
                mutable=True
            )

            # Wrap alt allele base counts on a MutableData object to allow
            # for predictive posterior checks
            alt = pm.MutableData(
                "alt", 
                vcf_fit.alt_bc.values, 
                dims = "sites"
            )

            # Probability of the query strain having the alternate allele
            alt_prob = pm.Beta(
                "alt_prob",
                alpha = 1,
                beta = 1,
                initval = "prior"
            )

            # Indicates if the query strain has the alt allele (1) or the 
            # ref allele (0)
            alleles = pm.Bernoulli(
                "alleles", 
                p = alt_prob, 
                dims = "sites",
                initval = "prior"
            )

            # Models the overdispersion for the depth of coverage of the 
            # query strain. Strong prior around 0.5 (approximates a Poisson
            # distribution). Use a normal distribution as prior as using a 
            # Beta distribution leads to inf likelihood (although it should
            # not)
            
            query_overdispersion = pm.Beta(
                "query_overdispersion", 
                alpha = self.overdispersion_bias, 
                beta = self.overdispersion_bias, 
                initval = 0.5
            )

            # Clip to max overdispersion
            query_overdispersion = pm.math.clip(
                query_overdispersion,
                1 - self.max_overdispersion,
                self.max_overdispersion
            )
            
            # Transform overdispersion
            query_overdispersion = query_overdispersion * 3 - 1.5
            query_overdispersion = query_coverage_estimate * (10**query_overdispersion)

            # Model the depth of coverage at each position in the query
            # strain as a Negative Binomial distribution
            query = pm.NegativeBinomial.dist(
                mu = query_coverage_estimate, 
                alpha = query_overdispersion
            )

            # Model the background allele depth as a uniform distribution
            # Add a 5 bp pad to ensure the likelihood does not explode
            max_bc = np.maximum(
                vcf.alt_bc.max(),
                vcf.ref_bc.max()
            ) + 5

            background = pm.Categorical.dist(
                p = np.ones(max_bc) / max_bc,
            )
            
            # Define the likelihood as a mixture, with weights given by the
            # alleles assigned to the query strain
            likelihood = pm.Mixture(
                "likelihood",
                w = pm.math.switch(
                    alleles[:,np.newaxis], 
                    pt.as_tensor(
                        [[1,0]]*len(vcf_fit)
                    ), 
                    pt.as_tensor(
                        [[0,1]]*len(vcf_fit)
                    )
                ),
                comp_dists = [query, background],
                observed = alt
            )

            # MCMC sampling
            logging.info(
                f"Starting MCMC sampling with {self.chains} chains, {self.draws} draws, {self.burn_in} \
burn-in draws, and {self.threads} threads. Random seed: {self.random_state}. Sampler: {self.engine}."
            )

            try:
                self.sampling_results = pm.sample(
                    chains=self.chains, 
                    draws=self.draws, 
                    random_seed=self.random_state, 
                    tune=self.burn_in, 
                    cores=self.threads,
                    nuts_sampler=self.engine,
                    initvals = {
                        "query_overdispersion": 0.5
                    }
                )

            except SamplingError as e:

                print(
                    model.debug(
                        verbose = True
                    )
                )

                print(
                    "Observed data:\n",
                    vcf_fit.alt_bc.values
                )

                raise e

            # Transform dispersion
            self.dist_params = self.__get_distribution_params(self.sampling_results)

            self.dist_params["query_overdispersion"] = query_coverage_estimate * (
                10**(
                    self.dist_params["query_overdispersion"] * 3 - 1.5
                )
            )
            
            # Predict for the full data 
            logging.debug(
                "Getting the MAP estimator for all data..."
            )
            logging.info
            alt_p = self.get_posterior(
                vcf, 
                self.sampling_results, 
                query_coverage_estimate,
            )
            
        return alt_p 

    def get_ll_ratio(
        self, 
        observed: pd.DataFrame, 
        sampling_results, 
        query_coverage_estimate: float
    ) -> pd.Series:

        # Build base count distributions for true and false variants based on the MCMC results
        dist_query = pm.NegativeBinomial.dist(
            mu = query_coverage_estimate,
            alpha = self.dist_params["query_overdispersion"]
        ) 

        # Likelihood given alt allele depths
        logp_alt = np.nan_to_num(
            pm.logp(
                dist_query,
                observed["alt_bc"].values
            ).eval(),
            nan = -100,
            posinf = 0,
            neginf = -100
        )

        # Likelihood given reference allele depths
        logp_ref = np.nan_to_num(
            pm.logp(
                dist_query,
                observed["ref_bc"].values
            ).eval(),
            nan = -100,
            posinf = 0,
            neginf = -100
        )

        # Prior logodds
        log_odds = np.log(
            self.dist_params["alt_prob"] / (
                1-self.dist_params["alt_prob"]+1e-10
            )
        )

        return pd.Series(
            logp_alt - logp_ref, 
            index = observed.index
        ) 
    
    def query_cdf(
        self,
        observed: pd.DataFrame, 
        sampling_results, 
        query_coverage_estimate: float,
        column: str
    ) -> pd.Series:
        
        # Build base count distributions for query alleles based on the MCMC results
                
        # Get the CDF for each reference allele base count
        dist_query = pm.NegativeBinomial.dist(
            mu = query_coverage_estimate,
            alpha = self.dist_params["query_overdispersion"]
        )
        query_cdf = np.exp(
            np.maximum(       
                pm.logcdf(
                    dist_query, 
                    observed[column].values
                ).eval(),
                -1e6
            )
        )

        return pd.Series(
            query_cdf,
            index = observed.index
        )
    
    def get_posterior(
        self, 
        observed: pd.DataFrame, 
        sampling_results, 
        query_coverage_estimate: float,
    ) -> pd.Series:

        ll_ratio = self.get_ll_ratio(
            observed = observed,
            sampling_results = sampling_results,
            query_coverage_estimate = query_coverage_estimate
        )

        # Check which sites originate from the distribution 
        # (p > left_quantile and p < right_quantile)
        alternate_cdf = self.query_cdf(
            observed, 
            sampling_results,
            query_coverage_estimate,
            "alt_bc"
        )

        # NOTE: May want to check only if <95%
        alt_evidence = (
            alternate_cdf.gt(self.__quantiles[0]) & \
            alternate_cdf.lt(self.__quantiles[1])
        )

        # Exclude sites with an alt allele depth not originating from the
        # distribution of depths of coverage for the query strain 
        ll_ratio[~alt_evidence] = -1

        return ll_ratio
    
    def __get_distribution_params(
        self, 
        sampling_results
    ) -> dict:

        posterior = sampling_results["posterior"]
        return {
            k: float(posterior[k].mean(dim=["chain", "draw"]))
            for k in [
                "query_overdispersion",
                "alt_prob"
            ]
        }

    def fit(
        self, 
        vcf: pd.DataFrame, 
        query_coverage_estimate: float,
        downsampling: Union[int, float] = 1.0        
    ) -> pd.Series:

        # Probability of the query having the alternate allele, given the alternate allele base count
        self.prob = self.fit_mcmc(
            vcf,
            query_coverage_estimate = query_coverage_estimate,
            downsampling = downsampling
        )

        logging.info(
            f"Estimated parameters for the distribution of depths of coverage for the query strain and \
alleles in the query strain: {'; '.join([k+': '+str(v) for k,v in self.dist_params.items()])}."
        )

        return self.prob

    def filter(self, vcf: pd.DataFrame) -> pd.DataFrame:

        return vcf[self.prob.ge(1)]
    
    def fit_filter(
        self, 
        vcf: pd.DataFrame, 
        query_coverage_estimate: float, 
        downsampling: Union[int, float] = 1.0
    ) -> pd.DataFrame:

        self.fit(
            vcf = vcf,
            query_coverage_estimate = query_coverage_estimate,
            downsampling = downsampling
        )
        return self.filter(vcf)

    def __get_coordinates(
        self, 
    ) -> dict:
        
        return {
            "alleles": ["ref", "alt"]
        }
    
    def __downsample_vcf(
        self, 
        vcf: pd.DataFrame, 
        n_lines: Union[int, float]
    ) -> pd.DataFrame:

        if n_lines <= 1:
            n_lines = int(len(vcf) * n_lines)
        n_lines_ = np.minimum(
            len(vcf), 
            int(n_lines)
        )

        return vcf.sample(
            n_lines_, 
            replace = False, 
            random_state = self.random_state
        )
# %%

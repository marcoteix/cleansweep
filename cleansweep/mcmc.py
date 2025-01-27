#%%
from functools import partial
from typing import Literal, Union
import numpy as np
import pymc as pm
from dataclasses import dataclass
import pandas as pd
from pytensor.printing import Print
from typing_extensions import Self
import pytensor.tensor as pt
from numpy.typing import ArrayLike

@dataclass
class BaseCountFilter:

    chains: int = 10
    draws: int = 1000
    burn_in: int = 100
    random_state: int = 23
    bias: float = 0.5
    threads: Union[int, None] = 4
    engine: str = "pymc"
    n_components: int = 5
    concentration: Union[float, None] = None

    def fit_mcmc(
        self, 
        vcf: pd.DataFrame, 
        coverages: dict, 
        query: str, 
        query_coverage_estimate: float,
        background_coverage_estimate: float,
        downsampling: Union[int, float] = 1.0,
    ) -> pd.Series:

        # Downsample the VCF file
        vcf_fit = self.__downsample_vcf(
            vcf, 
            n_lines = downsampling
        )

        coordinates = self.__get_coordinates(
            coverages, 
        )
        self.query = query

        with pm.Model(coords=coordinates) as model:

            model.add_coord(
                "sites", 
                vcf_fit.index.to_list(), 
                mutable=True
            )
            alt = pm.MutableData(
                "alt", 
                vcf_fit.alt_bc.values, 
                dims = "sites"
            )

            alt_prob = pm.Beta(
                "alt_prob",
                alpha = 1,
                beta = 1
            )

            alleles = pm.Bernoulli(
                "alleles", 
                p = alt_prob, 
                dims="sites"
            )

            query_overdispersion = pm.Beta(
                "query_overdispersion", 
                alpha = 20, 
                beta = 20, 
                initval = 0.5
            )

            query = pm.NegativeBinomial.dist(
                n = query_coverage_estimate, 
                p = query_overdispersion
            )

            background_overdispersion = pm.Beta(
                "background_overdispersion", 
                alpha=20, 
                beta=20, 
                initval=0.5
            )
            zero_inflation = pm.Beta(
                "zero_inflation",
                alpha = 1,
                beta = 1
            )
            background = pm.ZeroInflatedNegativeBinomial.dist(
                n = background_coverage_estimate, 
                p = background_overdispersion,
                psi = zero_inflation
            )
            
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

            self.sampling_results = pm.sample(
                chains=self.chains, 
                draws=self.draws, 
                random_seed=self.random_state, 
                tune=self.burn_in, 
                cores=self.threads,
                nuts_sampler=self.engine
            )
            
            # Predict for the full data 
            alt_p = self.get_posterior(
                vcf, 
                self.sampling_results, 
                query_coverage_estimate,
                background_coverage_estimate
            )
            
        return alt_p                     
            
    def get_conditional_posterior(
        self, 
        observed: pd.DataFrame, 
        sampling_results, 
        query_coverage_estimate: float,
        background_coverage_estimate: float,
        column: str = "alt_bc"
    ) -> pd.Series:

        # Build base count distributions for true and false variants based on the MCMC results
        dist_params = self.__get_distribution_params(sampling_results)

        dist_query = pm.NegativeBinomial.dist(
            n = query_coverage_estimate,
            p = dist_params["query_overdispersion"]
        )
        query_logp = np.maximum(
            pm.logp(
                dist_query, 
                observed[column].values
            ).eval(),
            -1e6
        )

        dist_background = pm.ZeroInflatedNegativeBinomial.dist(
            n = background_coverage_estimate,
            p = dist_params["background_overdispersion"],
            psi = dist_params["zero_inflation"]
        )
        background_logp = np.maximum(
            pm.logp(
                dist_background, 
                observed[column].values
            ).eval(),
            -1e6
        )

        prior = dist_params["alt_prob"]
        if prior == "ref_bc":
            prior = 1 - prior
        
        norm = (
            np.exp(query_logp)*prior
        ) + (
            np.exp(background_logp)*(1-prior)
        )

        return pd.Series(
            np.exp(query_logp)*prior/norm, 
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
        dist_params = self.__get_distribution_params(sampling_results)        
        
        # Get the CDF for each reference allele base count
        dist_query = pm.NegativeBinomial.dist(
            n = query_coverage_estimate,
            p = dist_params["query_overdispersion"]
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
        background_coverage_estimate: float,
    ) -> pd.Series:

        prob = {
            k: self.get_conditional_posterior(
                observed,
                sampling_results,
                query_coverage_estimate,
                background_coverage_estimate,
                column = k
            ) for k in [
                "ref_bc",
                "alt_bc"
            ]
        }

        # Get the CDF for the reference allele under the distribution
        # of query allele depths
        reference_cdf = self.query_cdf(
            observed, 
            sampling_results,
            query_coverage_estimate,
            "ref_bc"
        )
        # Check which sites originate from the distribution 
        # (p > 0.025 and p < 0.975)
        ref_evidence = (
            reference_cdf.gt(.025) & \
            reference_cdf.lt(0.975)
        )

        # Now do the same for the alternate allele
        alternate_cdf = self.query_cdf(
            observed, 
            sampling_results,
            query_coverage_estimate,
            "alt_bc"
        )
        alt_evidence = (
            alternate_cdf.lt(0.95)
        )

        # How many sites have evidence for the reference allele but 
        # not the alternate allele?
        prop_ref_sites = (
            ref_evidence & \
            ~alt_evidence
        ).mean()

        # If more than 10%, exclude sites with ref_bc originating 
        # from the distribution of query allele depths
        #if prop_ref_sites > 0.05:
        prob["alt_bc"][~alt_evidence] = 0.0

        return prob["alt_bc"]

    def __get_distribution_params(
        self, 
        sampling_results
    ) -> dict:

        posterior = sampling_results["posterior"]
        return {
            k: float(posterior[k].mean(dim=["chain", "draw"]))
            for k in [
                "query_overdispersion", 
                "zero_inflation",
                "background_overdispersion",
                "alt_prob"
            ]
        }

    def fit(
        self, 
        vcf: pd.DataFrame, 
        coverages: dict, 
        query: str, 
        query_coverage_estimate: float,
        background_coverage_estimate: float,
        downsampling: Union[int, float] = 1.0        
    ) -> pd.Series:

        # Probability of the query having the alternate allele, given the alternate allele base count
        self.prob = self.fit_mcmc(
            vcf,
            coverages,
            query,
            query_coverage_estimate = query_coverage_estimate,
            background_coverage_estimate = background_coverage_estimate,
            downsampling = downsampling
        )

        return self.prob

    def filter(self, vcf: pd.DataFrame) -> pd.DataFrame:

        return vcf[self.prob.ge(self.bias)]
    
    def fit_filter(self, vcf: pd.DataFrame, coverages: dict, query: str) -> pd.DataFrame:

        self.fit(vcf, coverages=coverages, query=query)
        return self.filter(vcf)

    def __get_coordinates(
        self, 
        coverages: dict, 
    ) -> dict:
        
        return {
            "strains": [x for x in coverages], 
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

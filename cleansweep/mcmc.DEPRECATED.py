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

    def fit_mcmc(
        self, 
        vcf: pd.DataFrame, 
        coverages: dict, 
        query: str, 
        query_coverage_estimate: float,
        background_coverage_estimate: float,
        downsampling: Union[int, float] = 1.0
    ) -> pd.Series:

        # Downsample the VCF file
        vcf_fit = self.__downsample_vcf(vcf, n_lines=downsampling)

        coordinates = self.__get_coordinates(query, coverages, vcf_fit.alt_bc)
        self.query = query

        samples = vcf_fit[["ref_bc", "alt_bc"]] \
            .values \
            .flatten()
        is_alt = pt.as_tensor(
            np.hstack([0, 1] * len(vcf_fit))
        )[:, np.newaxis]
        sites = vcf_fit.index \
            .to_numpy() \
            .repeat(2)
        idx = np.arange(
            len(vcf_fit)
        ).repeat(2)

        ambiguity_factor = vcf_fit["filter"].eq("PASS").sum()/len(vcf_fit)
        ambiguity_factor = np.maximum(ambiguity_factor, 1/3)
        self.ambiguity_factor = ambiguity_factor

        with pm.Model(coords=coordinates) as model:

            model.add_coord("sites", vcf_fit.index, mutable=True)

            alleles = pm.Bernoulli(
                "alleles", 
                p = ambiguity_factor, 
                dims="sites"
            )[idx]

            # Query coverage parameters
            
            query_coverage = pm.Poisson(
                "query_coverage",
                mu = pm.math.round(
                    query_coverage_estimate
                )
            ) + 1
            """
            query_overdispersion = pm.Beta(
                "query_overdispersion", 
                alpha=10, beta=10, 
                initval=0.5
            )
            query_bc = pm.NegativeBinomial.dist(
                n = query_coverage, 
                p = query_overdispersion
            )
            query_coverage = pm.Deterministic(
                "query_coverage",
                pm.math.round(coverages[query]*ambiguity_factor)
            )
            """
            query_bc = pm.Poisson.dist(
                mu = query_coverage
            )

            # Background coverage parameters
            background_coverage = pm.Poisson(
                "background_coverage",
                mu = pm.math.round(
                    background_coverage_estimate
                )
            ) + 1
            background_overdispersion = pm.Beta(
                "background_overdispersion", 
                alpha=10, beta=10, 
                initval=0.5
            )
            zi = pm.Beta("zi", alpha=1, beta=1)
            background_bc = pm.ZeroInflatedNegativeBinomial.dist(
                psi = zi,
                n = background_coverage, 
                p = background_overdispersion
            )
            background_bc = pm.ZeroInflatedPoisson.dist(
                psi = zi,
                mu = background_coverage
            )

            # Define the mixture
            weights = pm.math.switch(
                pm.math.switch(
                    is_alt,
                    alleles[:, np.newaxis],
                    1-alleles[:, np.newaxis]
                ),
                pt.as_tensor([[1,0]]*len(vcf_fit)*2), 
                pt.as_tensor([[0,1]]*len(vcf_fit)*2)
            )

            # If evaluating an alternate allele base count, invert the weights 
            
            likelihood = pm.Mixture(
                "likelihood",
                w = weights,
                comp_dists = [query_bc, background_bc],
                observed = samples
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
                ambiguity_factor,
                "alt_bc"
            ) * (1 - self.get_posterior(
                vcf, 
                self.sampling_results, 
                ambiguity_factor,
                "ref_bc"
            ))
            
        return alt_p           
            
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
        alt_p = self.fit_mcmc(
            vcf, 
            coverages, 
            query, 
            downsampling = downsampling,
            query_coverage_estimate = query_coverage_estimate,
            background_coverage_estimate = background_coverage_estimate
        )

        self.prob = alt_p
        return self.prob
    
    def get_distribution_params(self, sampling_results) -> dict:

        posterior = sampling_results["posterior"]
        return {
            k: float(posterior[k].mean(dim=["chain", "draw"]))
            for k in [
                "query_coverage",
                "background_coverage",
                "background_overdispersion", 
                "zi"
            ]
        }

    def get_posterior(
        self, 
        observed: pd.DataFrame, 
        sampling_results, 
        ambiguity_factor: float,
        column: str
    ) -> pd.Series:

        # Build base count distributions for true and false variants based on the MCMC results
        dist_params = self.get_distribution_params(sampling_results)

        query_dist = pm.Poisson.dist(
            mu = dist_params["query_coverage"],
        )
        background_dist = pm.ZeroInflatedNegativeBinomial.dist(
            psi = dist_params["zi"],
            n = dist_params["background_coverage"],
            p = dist_params["background_overdispersion"]
        )
        background_dist = pm.ZeroInflatedPoisson.dist(
            psi = dist_params["zi"],
            mu = dist_params["background_coverage"]
        )

        # Likelihood of the query having the alt allele
        alt_logp = pm.logp(
            query_dist,
            observed[column].values
        ).eval() + 1e-10

        # Likelihood of the query having the reference allele
        ref_logp = pm.logp(
            background_dist,
            observed[column].values
        ).eval() + 1e-10
        
        norm = np.exp(alt_logp)*ambiguity_factor + np.exp(ref_logp)*(1-ambiguity_factor)
        
        return pd.Series(
            np.exp(alt_logp)*ambiguity_factor/norm, 
            index=observed.index
        )

    def filter(self, vcf: pd.DataFrame) -> pd.DataFrame:

        return vcf[self.prob.ge(self.bias)]
    
    def fit_filter(self, vcf: pd.DataFrame, coverages: dict, query: str) -> pd.DataFrame:

        self.fit(vcf, coverages=coverages, query=query)
        return self.filter(vcf)
    
    def get_allele_p(self, sampling_results) -> pd.Series:

        if not hasattr(self, "sampling_results"):
            raise RuntimeError("The model must be fitted prior to calling \"get_allele_p()\".")
        
        return sampling_results["posterior"]["alleles"] \
            .mean(dim=["draw", "chain"]) \
            .to_dataframe().alleles

    def __get_coordinates(self, query: str, coverages: dict, bc: pd.Series) -> dict:
        
        return {
            "strains": [x for x in coverages], 
            "alleles": ["ref", "alt"]
        }
    
    def __downsample_vcf(self, vcf: pd.DataFrame, n_lines: Union[int, float]) -> pd.DataFrame:

        if n_lines <= 1:
            n_lines = int(len(vcf) * n_lines)
        n_lines_ = np.minimum(len(vcf), int(n_lines))

        return vcf.sample(n_lines_, replace=False, random_state=self.random_state)
# %%

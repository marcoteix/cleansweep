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

    def fit_mcmc2(self, vcf: pd.DataFrame, coverages: dict, query: str, downsampling: Union[int, float] = 1.0) -> pd.Series:

        # Downsample the VCF file
        vcf_fit = self.__downsample_vcf(vcf, n_lines=downsampling)

        coordinates = self.__get_coordinates(query, coverages, vcf_fit.alt_bc)
        self.query = query

        ambiguity_factor = vcf_fit["filter"].eq("PASS").sum()/len(vcf_fit)
        ambiguity_factor = np.maximum(ambiguity_factor, 1/3)
        print(ambiguity_factor)

        with pm.Model(coords=coordinates) as model:

            model.add_coord("sites", vcf_fit.index.to_list(), mutable=True)
            alt = pm.MutableData("alt", vcf_fit.alt_bc.values, dims="sites")

            alleles = pm.Bernoulli("alleles", p=ambiguity_factor, dims="sites")
            query_coverage = pm.NegativeBinomial("query_coverage", n=coverages[query]*ambiguity_factor, p=.6)
            query_overdispersion = pm.Beta("query_overdispersion", alpha=20, beta=20, initval=0.5)
            query_bc = pm.NegativeBinomial.dist(n=query_coverage, p=query_overdispersion)

            overdispersion = pm.Beta("overdispersion", alpha=20, beta=20, initval=0.5)
            background_prior = pm.NegativeBinomial("background_prior", n=coverages[query]*1.5, p=0.6) + 1
            background = pm.NegativeBinomial.dist(n=background_prior, p=overdispersion)
            
            likelihood = pm.Mixture(
                "likelihood",
                w = pm.math.switch(alleles[:,np.newaxis], pt.as_tensor([[1,0]]*len(vcf_fit)), pt.as_tensor([[0,1]]*len(vcf_fit))),
                comp_dists = [query_bc, background],
                observed = alt
            )

            self.sampling_results = pm.sample(chains=self.chains, draws=self.draws, 
                random_seed=self.random_state, tune=self.burn_in, cores=self.threads,
                nuts_sampler=self.engine)
            
            # Predict for the full data 
            alt_p = self.get_posterior(vcf, self.sampling_results, ambiguity_factor)
            
        return alt_p           

    def fit_mcmc(self, vcf: pd.DataFrame, coverages: dict, query: str, downsampling: Union[int, float] = 1.0) -> pd.Series:

        # Downsample the VCF file
        vcf_fit = self.__downsample_vcf(vcf, n_lines=downsampling)

        coordinates = self.__get_coordinates(query, coverages, vcf_fit.alt_bc)
        self.query = query

        ambiguity_factor = vcf_fit["filter"].eq("PASS").sum()/len(vcf_fit)
        ambiguity_factor = np.maximum(ambiguity_factor, 1/3)

        with pm.Model(coords=coordinates) as model:

            model.add_coord("sites", vcf_fit.index.to_list(), mutable=True)
            samples = pm.MutableData("samples", vcf_fit.alt_bc.values, dims="sites")

            alleles = pm.Bernoulli("alleles", p=ambiguity_factor, dims="sites")
            query_coverage = pm.Poisson("query_coverage", mu=coverages[query]*ambiguity_factor)
            query_overdispersion = pm.Beta("query_overdispersion", alpha=1, beta=1, initval=0.5)
            query_bc = pm.NegativeBinomial("query_bc", n=query_coverage, 
                p=query_overdispersion, dims="sites")

            zi_prior = pm.Beta("zi_prior", alpha=1, beta=1, initval=0.5)
            overdispersion = pm.Beta("overdispersion", alpha=1, beta=1, initval=0.5)
            background_prior = pm.Poisson("background_prior", mu=coverages[query]) + 1
            background = pm.ZeroInflatedNegativeBinomial(
                "background", psi=zi_prior, n=background_prior, 
                p=overdispersion, dims="sites")
            
            n_reads = alleles*query_bc + (1-alleles)*background

            # Errors are normally distributed
            #error_sd = pm.Gamma("error_sd", alpha=1, beta=5) + .01
            #likelihood = pm.Normal("likelihood", mu=n_reads, sigma=error_sd, observed=samples)

            likelihood = pm.Poisson("likelihood", mu=n_reads, observed=samples)

            self.sampling_results = pm.sample(chains=self.chains, draws=self.draws, 
                random_seed=self.random_state, tune=self.burn_in, cores=self.threads,
                nuts_sampler=self.engine)
            
            # Predict for the full data 
            alt_p = self.get_posterior(vcf, self.sampling_results, ambiguity_factor)
            
        return alt_p
            
    def fit(self, vcf: pd.DataFrame, coverages: dict, query: str, downsampling: Union[int, float] = 1.0) -> pd.Series:

        # Probability of the query having the alternate allele, given the alternate allele base count
        alt_p = self.fit_mcmc(vcf, coverages, query, downsampling=downsampling)

        self.prob = alt_p
        return self.prob
    
    def __get_distribution_params(self, sampling_results) -> dict:

        posterior = sampling_results["posterior"]
        return {
            k: float(posterior[k].mean(dim=["chain", "draw"]))
            for k in ["query_coverage", "query_overdispersion", "zi_prior",
                "overdispersion", "background_prior"]
        }
    
    def get_posterior2(self, observed: pd.DataFrame, sampling_results, ambiguity_factor: float) -> pd.Series:

        # Build base count distributions for true and false variants based on the MCMC results
        dist_params = self.__get_distribution_params(sampling_results)

        dist_query = pm.NegativeBinomial.dist(
            n = dist_params["query_coverage"],
            p = dist_params["query_overdispersion"]
        )
        query_logp = pm.logp(dist_query, observed.alt_bc.values).eval()

        dist_background = pm.NegativeBinomial.dist(
            n = dist_params["background_prior"],
            p = dist_params["overdispersion"]
        )
        background_logp = pm.logp(dist_background, observed.alt_bc.values).eval()

        prior = ambiguity_factor
        
        norm = np.exp(query_logp)*prior + np.exp(background_logp)*(1-prior)

        return pd.Series(np.exp(query_logp)*prior/norm, index=observed.index)
    
    def get_posterior(self, observed: pd.DataFrame, sampling_results, ambiguity_factor: float) -> pd.Series:

        # Build base count distributions for true and false variants based on the MCMC results
        dist_params = self.__get_distribution_params(sampling_results)

        dist_query = pm.NegativeBinomial.dist(
            n = dist_params["query_coverage"],
            p = dist_params["query_overdispersion"]
        )
        query_logp = pm.logp(dist_query, observed.alt_bc.values).eval()

        dist_background = pm.ZeroInflatedNegativeBinomial.dist(
            psi = dist_params["zi_prior"],
            n = dist_params["background_prior"],
            p = dist_params["overdispersion"]
        )
        background_logp = pm.logp(dist_background, observed.alt_bc.values).eval()

        prior = ambiguity_factor
        
        norm = np.exp(query_logp)*prior + np.exp(background_logp)*(1-prior)

        return pd.Series(np.exp(query_logp)*prior/norm, index=observed.index)
    
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

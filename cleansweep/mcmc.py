from functools import partial
from typing import Union
import numpy as np
import pymc as pm
from dataclasses import dataclass
import pandas as pd
from pytensor.printing import Print
from typing_extensions import Self
import pytensor.tensor as pt

@dataclass
class BaseCountFilter:

    chains: int = 10
    draws: int = 1000
    burn_in: int = 100
    random_state: int = 23
    bias: float = 0.5
    threads: Union[int, None] = 4
    engine: str = "pymc"

    def fit(self, vcf: pd.DataFrame, coverages: dict, query: str) -> pd.DataFrame:

        coordinates = self.__get_coordinates(coverages, vcf.alt_bc)
        self.query = query

        samples = vcf[["alt_bc", "ref_bc"]].values.flatten()
        is_ref = pt.as_tensor(np.hstack([np.array([0,1]*len(vcf)).reshape(-1,1)]*len(coverages)))
        sample_ids = np.repeat(np.arange(vcf.shape[0]), 2)

        coverages_pt = pt.as_tensor(list(coverages.values()))

        with pm.Model(coords=coordinates) as model:

            # Alleles indicates if each strain has the alternate (1) or reference (0) allele
            # at each site. Beta priors for the probability of each strain having the alternate 
            # allele
            allele_priors = pm.Beta("allele_priors", alpha=1, beta=1, dims="strains")
            alleles = pm.Bernoulli("alleles", p=allele_priors, dims=["sites", "strains"])
            # Define the binomial distribution parameters
            nb_mu = pm.Poisson("nb_mu", mu=coverages_pt, dims="strains")
            nb_alpha = pm.Beta("nb_alpha", alpha=1, beta=1)
            # Define the negative binomial for the alt allele
            alt_trials = pm.math.sum(nb_mu*pm.math.abs((is_ref-alleles[sample_ids,:])), axis=1) + 1
            pm.NegativeBinomial("nb", n=alt_trials, p=nb_alpha, observed=samples)

            self.sampling_results = pm.sample(chains=self.chains, draws=self.draws, 
                random_seed=self.random_state, tune=self.burn_in, cores=self.threads,
                nuts_sampler=self.engine)

        # Return the probabilities of the query having the alternate allele
        return self.get_allele_p(query)
    
    def filter(self, vcf: pd.DataFrame) -> pd.DataFrame:

        return vcf[self.get_allele_p(self.query).ge(self.bias)]
    
    def fit_filter(self, vcf: pd.DataFrame, coverages: dict, query: str) -> pd.DataFrame:

        self.fit(vcf, coverages=coverages, query=query)
        return self.filter(vcf)
    
    def get_allele_p(self, strain: str) -> pd.DataFrame:

        if not hasattr(self, "sampling_results"):
            raise RuntimeError("The model must be fitted prior to calling \"get_allele_p()\".")
        
        return self.sampling_results["posterior"]["alleles"] \
            .mean(dim=["draw", "chain"]) \
            .sel({"strains":strain}) \
            .to_dataframe().alleles

    def __get_coordinates(self, coverages: dict, bc: pd.Series) -> dict:
        
        return {
            "strains": [x for x in coverages], 
            "alleles": ["ref", "alt"],
            "sites": bc.index.to_list()
        }
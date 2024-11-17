#%%
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

        coordinates = self.__get_coordinates(query, coverages, vcf.alt_bc)
        self.query = query

        samples = vcf[["alt_bc", "ref_bc"]].values.flatten()
        is_ref = pt.as_tensor(np.hstack([np.array([0,1]*len(vcf))]))
        sample_ids = np.repeat(np.arange(vcf.shape[0]), 2)

        coverages_pt = pt.as_tensor(list(coverages.values()))
        bgd_coverages = pt.as_tensor([v for k,v in coverages.items() if k!= query])

        with pm.Model(coords=coordinates) as model:

            # Alleles indicates if each strain has the alternate (1) or reference (0) allele
            # at each site. Beta priors for the probability of each strain having the alternate 
            # allele
            query_allele_prior = pm.Beta("query_allele_prior", alpha=1, beta=1)
            query_allele = pm.Bernoulli("query_allele", p=query_allele_prior, dims="sites")[sample_ids]

            bgd_aln_prior = pm.Beta("bgd_aln_prior", alpha=1, beta=10, dims="strains")
            bgd_aln = pm.Bernoulli("bgd_aln", p=bgd_aln_prior, dims=["sites", "strains"])[sample_ids,:]
            bgd_allele = pm.Bernoulli("bgd_allele", p=0.5, dims=["sites", "strains"])[sample_ids,:]

            # Effective depth of coverage of the background and query strains
            query_coverage = pm.Poisson("query_coverage", mu=coverages[query])
            bgd_coverage = pm.Poisson("bgd_coverage", mu=bgd_coverages, dims="strains")

            # Total expected number of reads
            query_n_reads = (is_ref*query_allele+(1-is_ref)*(1-query_allele))*query_coverage
            bkg_switch = (is_ref[:,np.newaxis]*bgd_allele+(1-is_ref[:,np.newaxis])*(1-bgd_allele))
            bkg_n_reads = pm.math.sum(bgd_aln*bkg_switch*bgd_coverage[np.newaxis,:], axis=1)
            n_reads = query_n_reads + bkg_n_reads + 1
            
            pm.Poisson("likelihood", mu=n_reads, observed=samples)

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
        
        return self.sampling_results["posterior"]["query_allele"] \
            .mean(dim=["draw", "chain"]) \
            .to_dataframe().query_allele

    def __get_coordinates(self, query: str, coverages: dict, bc: pd.Series) -> dict:
        
        return {
            "strains": [x for x in coverages if x!= query], 
            "alleles": ["ref", "alt"],
            "sites": bc.index.to_list()
        }
# %%

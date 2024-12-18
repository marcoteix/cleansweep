#%%
from dataclasses import dataclass
from typing import Union
from cleansweep.coverage import CoverageFilter
from cleansweep.io import FilePath
from cleansweep.mcmc import BaseCountFilter
import pandas as pd
import joblib
from copy import deepcopy

@dataclass
class VCFFilter:

    random_state: int = 23

    def fit(
        self, 
        vcf: pd.DataFrame, 
        coverages: pd.Series,
        query_name: str, 
        *,
        coverage_min_p = 0.1,
        min_alt_bc: int = 10,
        min_ref_bc: int = 10,
        min_ambiguity: float = 0.0,
        downsampling: Union[int, float] = 1.0,
        chains: int = 5,
        draws: int = 10000,
        burn_in: int = 1000,
        bias: float = 0.5,
        threads: int = 4,
        engine: str = "pymc",
        coverage_filter_params: dict = {}
    ) -> pd.Series:

        # Coverage of the query strain
        self.query_coverage = self.__get_query_coverage(coverages=coverages, query_name=query_name)

        # Automatically fail all variants with an alternate allele base count < min_alt_bc
        # Exclude variants with a reference allele base count < min_ref_bc from the estimation
        vcf = vcf.assign(
            low_alt_bc = vcf.alt_bc.lt(min_alt_bc),
            low_ref_bc = vcf.ref_bc.lt(min_ref_bc)
        )

        # Fit and apply the coverage-based filter
        self.coverage_filter = CoverageFilter(random_state=self.random_state)
        vcf = self.coverage_filter.fit(vcf, self.query_coverage, 
            p_threshold=coverage_min_p, **coverage_filter_params)
        filtered_vcf = self.coverage_filter.filter(vcf)

        # Check if MCMC is needed
        skip_mcmc = self.__lt_min_ambiguity(vcf, min_ambiguity=min_ambiguity)
        if skip_mcmc:
            p_alt = vcf["filter"].eq("PASS").replace({True: 1.0, False: 0.0})
        else:
            self.basecount_filter = BaseCountFilter(chains=chains, draws=draws, burn_in=burn_in,
                bias=bias, threads=threads, engine=engine)
            # Fit and get the probabilities of the query having the alternate allele
            filtered_vcf = filtered_vcf[~filtered_vcf.low_ref_bc]
            p_alt = self.basecount_filter.fit(vcf=filtered_vcf, coverages=coverages.to_dict(), 
                query=query_name, downsampling=downsampling)
        # Join the probabilities with the VCF DataFrame
        vcf = vcf.join(p_alt.rename("p_alt"))
        return self.__add_filter_tag(vcf=vcf, bias=bias)

    def save_samples(self, path: FilePath) -> None:
        if hasattr(self, "basecount_filter"):
            joblib.dump(
                deepcopy(self.basecount_filter.sampling_results), 
                path,
                compress=5
            )

    def save(self, path: FilePath) -> None:

        joblib.dump(
            deepcopy(self),
            path,
            compress=5
        )

    def __add_filter_tag(self, vcf: pd.DataFrame, bias: float) -> pd.DataFrame:

        vcf = vcf.assign(cleansweep_filter=vcf.p_alt.ge(bias).replace(
            {True: "PASS", False: "FAIL"}))
        vcf.loc[vcf.low_alt_bc, "cleansweep_filter"] = "FAIL"
        vcf.loc[vcf.low_ref_bc, "cleansweep_filter"] = "PASS"
        vcf.loc[:, "cleansweep_filter"] = vcf.cleansweep_filter.fillna("HighCov")

        return vcf

    def __get_query_coverage(self, coverages: pd.Series, query_name: str) -> float:

        if not query_name in coverages.index:
            raise ValueError(f"No coverage information found for the query strain ({query_name}). \
Got coverage information for the strains {', '.join(coverages.index.to_list())}.")
        return coverages[query_name]
    
        
    def __lt_min_ambiguity(self, vcf: pd.DataFrame, min_ambiguity: float) -> bool:

        pct_ambiguous =  vcf["filter"].ne("PASS").sum()/len(vcf)
        if pct_ambiguous < min_ambiguity:
            print(f"Fewer than {min_ambiguity*100:.0f}% of variants are ambiguous ({pct_ambiguous*100:.0f}%). \
Skipping the base count filter and using the Pilon filters...")
            return True
        else: return False
# %%

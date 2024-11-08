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
        chains: int = 5,
        draws: int = 10000,
        burn_in: int = 1000,
        bias: float = 0.5,
        threads: int = 4,
        coverage_filter_params: dict = {}
    ) -> pd.Series:

        # Coverage of the query strain
        self.query_coverage = self.__get_query_coverage(coverages=coverages, query_name=query_name)

        # Fit and apply the coverage-based filter
        self.coverage_filter = CoverageFilter(random_state=self.random_state)
        vcf = self.coverage_filter.fit(vcf, self.query_coverage, **coverage_filter_params)
        filtered_vcf = self.coverage_filter.filter(vcf)

        self.basecount_filter = BaseCountFilter(chains=chains, draws=draws, burn_in=burn_in,
            bias=bias, threads=threads)
        # Fit and get the probabilities of the query having the alternate allele
        p_alt = self.basecount_filter.fit(vcf=filtered_vcf, coverages=coverages.to_dict(), 
            query=query_name)
        # Join the probabilities with the VCF DataFrame
        vcf = vcf.join(p_alt.rename("p_alt"))
        return self.__add_filter_tag(vcf=vcf)

    def save_samples(self, path: FilePath) -> None:

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

    def __add_filter_tag(self, vcf: pd.DataFrame) -> pd.DataFrame:

        vcf = vcf.assign(cleansweep_filter=vcf.p_alt.ge(self.bias)).replace(
            {True: "PASS", False: "FAIL"})
        vcf.loc[:, "cleansweep_filter"] = vcf.cleansweep_filter.fillna("HighCov")

        return vcf

    def __get_query_coverage(self, coverages: pd.Series, query_name: str) -> float:

        if not query_name in coverages.index:
            raise ValueError(f"No coverage information found for the query strain ({query_name}). \
Got coverage information for the strains {', '.join(coverages.index.to_list())}.")
        return coverages[query_name]
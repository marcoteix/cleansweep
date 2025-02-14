#%%
from dataclasses import dataclass
from typing import Union, Iterable
import numpy as np
from cleansweep.coverage import CoverageFilter, CoverageEstimator
from cleansweep.io import FilePath
from cleansweep.mcmc import BaseCountFilter
from cleansweep.typing import File, Directory
from cleansweep.augment import AugmentVariantCalls
from pathlib import Path
import pandas as pd
import joblib
import shutil
from copy import deepcopy

NUCMER_SNPS_HEADER = [
    "pos",
    "ref",
    "alt",
    "query_pos",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "ref_id",
    "query_id"
]

class NucmerSNPFilter:

    def __init__(self):
        pass 

    def read_snps(
        self,
        files: Iterable[File]
    ) -> pd.DataFrame:
        
        return pd.concat(
            [
                pd.read_table(
                    file,
                    skiprows = 4,
                    names = NUCMER_SNPS_HEADER
                )
                for file in files
            ]
        ).reset_index()
    
    def filter(
        self,
        vcf: pd.DataFrame,
        nucmer_snps: Iterable[File]
    ) -> pd.DataFrame:
        
        # Load SNPs
        blacklist = self.read_snps(
            nucmer_snps
        )

        # Fail SNPs in the input VCF also present in the set of nucmer SNPs
        return vcf.assign(
            snp_filter = vcf.pos \
                .isin(
                    blacklist.pos
                ).replace(
                    {
                        True: "FAIL",
                        False: "PASS"
                    }
                )
        )


@dataclass
class VCFFilter:

    random_state: int = 23

    def fit(
        self, 
        query: str,
        vcf: File, 
        nucmer_snps: Iterable[File],
        tmp_dir: Directory,
        *,
        n_coverage_sites: int = 100000,
        min_depth: int = 0,
        min_alt_bc: int = 0,
        min_ref_bc: int = 0,
        max_overdispersion: float = 0.55,
        downsampling: Union[int, float] = 1.0,
        chains: int = 5,
        draws: int = 10000,
        burn_in: int = 1000,
        power: float = 0.975,
        threads: int = 5,
        engine: str = "pymc"
    ) -> pd.Series:   
    
        # Step 1: estimate the coverage of the background strain

        self.coverage_estimator = CoverageEstimator(
            random_state = self.random_state
        )

        self.query_coverage = self.coverage_estimator \
            .fit(
                vcf = vcf,
                n_lines = n_coverage_sites,
                min_depth = min_depth
            )
        
        # Step 2: include sites with a non-reference base count > alpha regardless
        # of if these were called as variants according to Pilon

        augment = AugmentVariantCalls()
        augment_min_alt_bc = augment.estimate_min_alt_bc(
            self.query_coverage,
            alpha = 0.01,
            overdispersion = max_overdispersion
        )

        # Path to the augmented VCF
        Path(tmp_dir).mkdir(
            exist_ok = True,
            parents = True
        )

        augmented_vcf = Path(tmp_dir) \
            .joinpath(
                "cleansweep.augmented.vcf"
            )
        vcf = augment.augment(
            vcf = vcf,
            query = query,
            min_alt_bc = augment_min_alt_bc,
            output = augmented_vcf
        )

        # Delete tmp directory
        shutil.rmtree(
            tmp_dir
        )

        # Step 3: exclude low coverage variants and variants with an alternate allele 
        # base count < min_alt_bc. Pass variants with a reference allele base count < 
        # min_ref_bc and ignore them on the following steps

        vcf = vcf.assign(
            low_cov = vcf.depth.lt(min_depth),
            low_alt_bc = vcf.alt_bc.lt(min_alt_bc),
            low_ref_bc = vcf.ref_bc.lt(min_ref_bc)
        )

        # Step 4: exclude SNPs between the reference sequences (from nucmer alignments)

        self.nucmer_filter = NucmerSNPFilter()
        vcf = self.nucmer_filter.filter(
            vcf = vcf,
            nucmer_snps = nucmer_snps
        )

        # Step 5: filter SNPs based on allele depths
        
        self.basecount_filter = BaseCountFilter(
            chains=chains, 
            draws=draws, 
            burn_in=burn_in,
            power=power, 
            threads=threads, 
            engine=engine
        )
        
        # Fit and get the probabilities of the query having the alternate allele
        mcmc_vcf = vcf[
            ~vcf.low_ref_bc & \
            ~vcf.low_cov & \
            vcf.snp_filter.eq("PASS") 
        ]

        p_alt = self.basecount_filter.fit(
            vcf = mcmc_vcf, 
            downsampling = downsampling,
            query_coverage_estimate = self.query_coverage,
        )

        # Join the probabilities with the full VCF DataFrame
        vcf = vcf.join(
            p_alt.rename("p_alt")
        )
        return self.__add_filter_tag(
            vcf = vcf,
            bias = 0.5
        )


    def fit2(
        self, 
        vcf: pd.DataFrame, 
        coverages: pd.Series,
        query_name: str, 
        downsampled_vcf: File,
        nucmer_snps: Iterable[File],
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
        self.query_coverage = self.__get_query_coverage(
            coverages = coverages, 
            query_name = query_name
        )

        # Automatically fail all variants with an alternate allele base count < min_alt_bc
        # Exclude variants with a reference allele base count < min_ref_bc from the estimation
        vcf = vcf.assign(
            low_alt_bc = vcf.alt_bc.lt(min_alt_bc),
            low_ref_bc = vcf.ref_bc.lt(min_ref_bc)
        )

        # Filter SNPs based on nucmer alignments
        self.nucmer_filter = NucmerSNPFilter()
        vcf = self.nucmer_filter.filter(
            vcf,
            nucmer_snps
        )

        # Fit and apply the coverage-based filter
        self.coverage_filter = CoverageFilter(
            random_state = self.random_state
        )
        vcf = self.coverage_filter.fit(
            vcf, 
            self.query_coverage, 
            p_threshold = coverage_min_p, 
            **coverage_filter_params
        )
        filtered_vcf = vcf #self.coverage_filter.filter(vcf)

        # Check if MCMC is needed
        skip_mcmc = self.__lt_min_ambiguity(
            filtered_vcf, 
            min_ambiguity = min_ambiguity
        )

        if skip_mcmc:

            p_alt = filtered_vcf["filter"] \
                .eq("PASS") \
                .replace(
                    {True: 1.0, False: 0.0}
                )
            
        else:

            # Estimate the query and background coverages 
            self.coverage_estimator = CoverageEstimator(
                random_state = self.random_state
            )

            depths = self.coverage_estimator \
                .read(downsampled_vcf)
            
            # Exclude NaNs
            depths = depths[~np.isnan(depths)]
            
            query_coverage, background_coverage = self.coverage_estimator \
                .estimate(depths)

            self.basecount_filter = BaseCountFilter(
                chains=chains, 
                draws=draws, 
                burn_in=burn_in,
                bias=bias, 
                threads=threads, 
                engine=engine
            )
            
            # Fit and get the probabilities of the query having the alternate allele
            mcmc_vcf = filtered_vcf[
                ~filtered_vcf.low_ref_bc & \
                filtered_vcf.snp_filter.eq("PASS") 
                #filtered_vcf.coverage_filter.eq("PASS")
            ]

            p_alt = self.basecount_filter.fit(
                vcf = mcmc_vcf, 
                coverages = coverages.to_dict(), 
                query = query_name, 
                downsampling = downsampling,
                query_coverage_estimate = query_coverage,
                background_coverage_estimate = background_coverage
            )

        # Join the probabilities with the VCF DataFrame
        vcf = vcf.join(
            p_alt.rename("p_alt")
        )
        return self.__add_filter_tag(
            vcf = vcf, 
            bias = bias
        )

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

    def __add_filter_tag(
        self, 
        vcf: pd.DataFrame, 
        bias: float
    ) -> pd.DataFrame:

        vcf = vcf.assign(
            cleansweep_filter = vcf.p_alt \
                .ge(bias) \
                .replace(
                    {
                        True: "PASS", 
                        False: "FAIL"
                    }
                )
            )
        vcf.loc[
            vcf.low_alt_bc,
            "cleansweep_filter"
        ] = "LowAltBC"
        vcf.loc[
            vcf.snp_filter.eq("FAIL"),
            "cleansweep_filter"
        ] = "RefVar"
        vcf.loc[
            vcf.low_ref_bc, 
            "cleansweep_filter"
        ] = "PASS"
        vcf.loc[
            vcf.low_cov,
            "cleansweep_filter"
        ] = "LowCov"

        return vcf

    def __get_query_coverage(
        self, 
        coverages: pd.Series, 
        query_name: str
    ) -> float:

        if not query_name in coverages.index:
            raise ValueError(f"No coverage information found for the query strain ({query_name}). \
Got coverage information for the strains {', '.join(coverages.index.to_list())}.")
        return coverages[query_name]
    
        
    def __lt_min_ambiguity(
        self, 
        vcf: pd.DataFrame, 
        min_ambiguity: float
    ) -> bool:

        pct_ambiguous =  vcf["filter"].ne("PASS").sum()/len(vcf)
        if pct_ambiguous < min_ambiguity:
            print(f"Fewer than {min_ambiguity*100:.0f}% of variants are ambiguous ({pct_ambiguous*100:.0f}%). \
Skipping the base count filter and using the Pilon filters...")
            return True
        else: return False
# %%

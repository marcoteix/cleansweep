#%%
from dataclasses import dataclass
from typing import List, Union, Iterable
import numpy as np
from cleansweep.coverage import CoverageFilter, CoverageEstimator
from cleansweep.io import FilePath
from cleansweep.mcmc import BaseCountFilter
from cleansweep.typing import File, Directory
from cleansweep.augment import AugmentVariantCalls
from cleansweep.nucmer import NUCMER_SNPS_HEADER
from pathlib import Path
import pandas as pd
import joblib
import shutil
import logging
from copy import deepcopy
import logging

class NucmerSNPFilter:

    def __init__(self):
        pass 
    
    def filter(
        self,
        vcf: pd.DataFrame,
        nucmer_snps: pd.DataFrame
    ) -> pd.DataFrame:

        logging.debug( 
            f"Found {len(nucmer_snps)} SNPs among reference sequences."
        )

        # Fail SNPs in the input VCF also present in the set of nucmer SNPs
        return vcf.assign(
            snp_filter = vcf.pos \
                .isin(
                    nucmer_snps.pos
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
        gaps: pd.DataFrame,
        nucmer_snps: pd.DataFrame,
        tmp_dir: Directory,
        *,
        n_coverage_sites: int = 100000,
        min_depth: int = 0,
        min_alt_bc: int = 0,
        min_ref_bc: int = 0,
        max_overdispersion: float = 0.1,
        downsampling: Union[int, float] = 1.0,
        chains: int = 5,
        draws: int = 10000,
        burn_in: int = 1000,
        power: float = 0.975,
        threads: int = 5,
        engine: str = "pymc",
        overdispersion_bias: int = 1
    ) -> pd.Series:   
    
        # Step 1: estimate the coverage of the background strain

        logging.info(f"Estimating the mean depth of coverage for {query}...")

        self.coverage_estimator = CoverageEstimator(
            random_state = self.random_state
        )

        self.query_coverage = self.coverage_estimator \
            .fit(
                vcf = vcf,
                query = query,
                gaps = gaps,
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

        logging.debug(
            "Finding low coverage variants and those with low alt and ref base counts"
        )

        vcf = vcf.assign(
            low_cov = vcf.depth.lt(min_depth),
            low_alt_bc = vcf.alt_bc.lt(min_alt_bc),
            low_ref_bc = vcf.ref_bc.lt(min_ref_bc)
        )

        logging.info(
            f"Found {vcf.low_cov.sum()} low coverage variants, {vcf.low_alt_bc.sum()} variants \
with low alternate base counts, and {vcf.low_ref_bc.sum()} variants with low reference base counts."
        )

        # Step 4: exclude SNPs between the reference sequences (from nucmer alignments)

        self.nucmer_filter = NucmerSNPFilter()
        vcf = self.nucmer_filter.filter(
            vcf = vcf,
            nucmer_snps = nucmer_snps
        )

        logging.info( 
            f"Filtered out {vcf.snp_filter.ne('PASS').sum()} variants also found in the \
reference sequences."
        )

        # Step 5: filter SNPs based on allele depths
        
        self.basecount_filter = BaseCountFilter(
            chains=chains, 
            draws=draws, 
            burn_in=burn_in,
            power=power, 
            threads=threads, 
            engine=engine,
            overdispersion_bias=overdispersion_bias
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
        vcf = vcf.drop(
            columns = "p_alt",
            errors = "ignore"
            ).join(
                p_alt.rename("p_alt")
            )
        
        return self.__add_filter_tag(
            vcf = vcf,
            bias = 0
        )

    def save_samples(self, path: FilePath) -> None:
        
        if hasattr(self, "basecount_filter"):
            
            logging.debug(
                f"Saving MCMC sampling results to {str(path)}..."
            )

            joblib.dump(
                deepcopy(self.basecount_filter.sampling_results), 
                path,
                compress=5
            )
        
        else:
            raise RuntimeError("VCFFilter has no MCMC sampling results.")

    def save(self, path: FilePath) -> None:

        logging.debug(
            f"Saving CleanSweep filter as {str(path)}..."
        )

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
# %%

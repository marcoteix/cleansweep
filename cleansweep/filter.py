#%%
from dataclasses import dataclass
from typing import Union
import numpy as np
from cleansweep.coverage import CoverageEstimator
from cleansweep.io import FilePath
from cleansweep.mcmc import AlleleDepthFilter
from cleansweep.typing import File, Directory
from cleansweep.augment import AugmentVariantCalls
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
        max_dispersion: float = 0.7,
        downsampling: Union[int, float] = 1.0,
        chains: int = 5,
        draws: int = 10000,
        burn_in: int = 1000,
        power: float = 0.975,
        threads: int = 5,
        dispersion_bias: int = 1,
        alt_allele_p_step_size: float = 0.1,
        dispersion_step_size: float = 0.1,
        allele_step_size: float = 0.1,
        min_acceptance_rate: float = 0.2,
        max_acceptance_rate: float = 0.6,
        adaptive_step: float = 0.1,
        block_size: float = 0.05,
        use_mle: Union[bool, None] = None
    ) -> pd.Series:   
        
        # Keep track of the passed options
        self.__opts = {
            "Number of sites used to estimate the mean depth of coverage": n_coverage_sites,
            "Minimum depth of coverage": min_depth,
            "Minimum alternate allele depth": min_alt_bc,
            "Minimum reference allele depth": min_ref_bc,
            "Maximum expected dispersion": max_dispersion,
            "Fraction of candidate variants used in MCMC": downsampling,
            "Number of MCMC chains": chains,
            "Number of MCMC draws": draws,
            "Number of MCMC burn-in draws": burn_in,
            "Number of threads": threads,
            "Dispersion bias (alpha and beta)": dispersion_bias,
            "Alternate allele probability step size": alt_allele_p_step_size,
            "Dispersion step size": dispersion_step_size,
            "Allele step size": allele_step_size,
            "Minimum target acceptance rate": min_acceptance_rate,
            "Maximum target acceptance rate": max_acceptance_rate,
            "Adaptive step size coefficient": adaptive_step,
            "MCMC allele update block size": block_size,
            "Force MLE?": use_mle,
            "Random state": self.random_state
        }
    
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
            overdispersion = max_dispersion
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
        
        self.basecount_filter = AlleleDepthFilter(
            query_coverage = self.query_coverage,
            samples = draws,
            burnin = burn_in,
            chains = chains,
            dispersion_bias = dispersion_bias,
            alt_allele_p_proposal_sd = alt_allele_p_step_size,
            dispersion_proposal_sd = dispersion_step_size,
            proposal_p = allele_step_size,
            min_acceptance_rate = min_acceptance_rate,
            max_acceptance_rate = max_acceptance_rate,
            adaptive_step = adaptive_step,
            step_size_range = 2,
            block_size = block_size,
            threads = threads,
            random_state = self.random_state,
            notebook = False
        )
        
        # Fit and get the probabilities of the query having the alternate allele
        mcmc_vcf = vcf[
            ~vcf.low_ref_bc & \
            ~vcf.low_cov & \
            vcf.snp_filter.eq("PASS") 
        ]

        # Fit model

        # If use_mle is True, force MLE estimator. If False, force 
        # estimates based on posterior probabilities. If None,
        # use MLE if the number of samples passed to MCMC is less
        # than the total number of variants
        n_mcmc_variants = self.__convert_downsampling(
            n_samples = downsampling,
            n_total = len(mcmc_vcf)
        )

        if use_mle is None:
            use_mle = (n_mcmc_variants != len(mcmc_vcf))

        self.__used_mle = use_mle

        if use_mle:
            # Select n_mcmc_variants
            self.basecount_filter.fit(
                mcmc_vcf.alt_bc.sample(
                    n_mcmc_variants,
                    replace = False
                )
            )
        else:
            # Use all variants
            self.basecount_filter.fit(mcmc_vcf.alt_bc)
        
        p_alt = self.basecount_filter.predict(
            mcmc_vcf.alt_bc,
            mcmc_vcf.ref_bc,
            power = power,
            use_mle = use_mle
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
                deepcopy(self.basecount_filter.to_dict()["posterior"]), 
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
            deepcopy(
                {
                    "opts": self.__opts,
                    "query_coverage": self.query_coverage,
                    "query_coverage_estimator": self.coverage_estimator,
                    "nucmer_filter": self.nucmer_filter,
                    "basecount_filter": self.basecount_filter.to_dict(),
                    "mle": self.__used_mle
                }
            ),
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
    
    def __convert_downsampling(self, n_samples: Union[int, float], n_total: int) -> int:
        """
        Converts the number of variants used for MCMC from a fraction
        of the total to a number.
        """

        if n_samples > 1:
            return np.maximum(int(n_samples), n_total)
        elif n_samples <= 0:
            return n_total 
        else:
            return np.maximum(int(n_samples*n_total), n_total)
# %%

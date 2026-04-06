import pandas as pd
import numpy as np
from typing import Tuple, Union
from cleansweep.typing import File
from warnings import warn
from scipy import stats as sps
from numpy.typing import ArrayLike
from dataclasses import dataclass
import logging
import pysam

@dataclass
class CoverageEstimator:

    random_state: int = 23

    def __post_init__(self):

        self.__rng = np.random.default_rng(
            self.random_state
        )

    def fit(
        self,
        vcf: File,
        query: str,
        gaps: pd.DataFrame,
        n_lines: int = 100000,
        min_depth: int = 0,
        **kwargs
    ) -> Tuple[float, Union[None, sps.rv_discrete]]:

        # Read depths from a VCF file
        depths = self.read(
            vcf = vcf,
            query = query,
            gaps = gaps,
            n_lines = n_lines
        )

        # Remove NaNs
        depths = depths[~np.isnan(depths)]

        if not len(depths):

            msg = f"Found no valid unaligned sites with available depths of coverage."
            logging.error(msg)
            raise ValueError(msg)

        # Remove low coverage sites
        depths = depths[depths >= min_depth]

        query_coverage, nbinom = self.estimate(
            depths,
            **kwargs
        )

        return query_coverage, nbinom

    def read(
        self,
        vcf: File,
        query: str,
        gaps: pd.DataFrame,
        n_lines: int = 100000
    ) -> ArrayLike:

        logging.debug(
            f"Downsampling {str(vcf)} to {n_lines} lines and extracting \
depths of coverage."
        )
        
        # Load VCF and downsample. Exctract depth of coverage at each 
        # position
        self.depths = self.downsample_vcf_depths(
            vcf = vcf,
            query = query,
            gaps = gaps,
            n_lines = n_lines
        )

        logging.debug(f"Got {len(self.depths)} valid depths of coverage.")

        return self.depths
    
    def estimate(
        self,
        depths: ArrayLike,
        **kwargs
    ) -> Tuple[float, Union[None, sps.rv_discrete]]:
        
        dist = None
        
        # Fit a Negative Binomial distribution to the data using MLE
        r, p = self._fit_nbinom_mle(depths)

        logging.debug(f"NB MLE estimates: r = {r:.4f}, p = {p:.4f}")

        dist = sps.nbinom(r, p)
        self.r = r
        self.p = p

        return np.median(depths), dist

    def downsample_vcf_depths(
        self,
        vcf: File,
        query: str,
        gaps: pd.DataFrame,
        n_lines: int = 100000,
    ) -> pd.DataFrame:
        
        gaps = gaps.reset_index()

        if not len(gaps):
            raise ValueError(
                f"Found no gaps. Cannot estimate the query depth of coverage."
            )
        
        # Load VCF file
        pysam_vcf = pysam.VariantFile(
            vcf,
            mode = "r"
        )

        # Loci that did not align to any background strain
        unaligned_loci = np.hstack(
            [
                np.arange(
                    x.start, 
                    (
                        x.end
                        if x.end > -1
                        else pysam_vcf.header.contigs[query].length
                    )
                )
                for _, x in gaps.iterrows()
            ]
        )

        logging.debug(f"Found {len(unaligned_loci)} unaligned loci for {query}.")

        # Select random positions
        selected_loci = self.__rng.choice(
            unaligned_loci,
            size = n_lines,
            replace = False
        )

        # Fetch VCF lines with pysam and extract depth of coverage
        depths = [
            self.__get_dp(
                pysam_vcf,
                query, i
            )
            for i in selected_loci
            if not self.__get_dp(
                pysam_vcf,
                query, i
            ) is None
        ]

        if not len(depths):

            # Selected positions not in the VCF file. Maybe the input
            # is a VCF with variant positions only or filtered some
            # other way. Warn the user and fall back on reading the 
            # entire file and selecting existing positions at random
            msg = "Failed to randomly select positions in the input \
VCF file. Please make sure the input file contains information for \
all loci. Attempting to select from existing lines in the VCF file. "
            warn(msg)
            logging.warning(msg)

            # Read existing positions for the query
            existing_depths = [
                x.info.get("DP")
                for x in pysam_vcf.fetch(query)
            ]

            depths = self.__rng.choice(
                existing_depths,
                size = np.minimum(
                    len(existing_depths),
                    n_lines
                ),
                replace = False
            )

        return np.array(depths)
    
    def __get_dp(
        self,
        record: pysam.VariantFile,
        query: str,
        i: int
    ) -> float:
        
        try:

            depth = next(
                record.fetch(
                    contig = query,
                    start = i,
                    stop = i + 1
                )
            ).info.get("DP")

        except StopIteration:

            depth = None 

        return depth
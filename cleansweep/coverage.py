from functools import partial
import random
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Collection, Callable, Any
from cleansweep.vcf import VCF, get_info_value
from cleansweep.typing import File
from warnings import warn
from scipy.stats import norm, multivariate_normal, poisson, rv_continuous
from numpy.typing import ArrayLike
from dataclasses import dataclass
import subprocess
import io
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
    ) -> float:

        # Read depths from a VCF file
        depths = self.read(
            vcf = vcf,
            query = query,
            gaps = gaps,
            n_lines = n_lines
        )

        # Remove NaNs
        depths = depths[~np.isnan(depths)]
        # Remove low coverage sites
        depths = depths[depths >= min_depth]

        query_coverage, _ = self.estimate(
            depths,
            **kwargs
        )

        return query_coverage

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
    ) -> Tuple[float, float]:
        
        return np.median(depths), None

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
            next(
                pysam_vcf.fetch(
                    contig = query,
                    start = i,
                    stop = i + 1
                )
            ).info.get("DP")
            for i in selected_loci
        ]

        return depths
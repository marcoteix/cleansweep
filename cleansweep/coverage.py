from functools import partial
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

@dataclass
class CoverageEstimator:

    random_state: int = 23

    def fit(
        self,
        vcf: File,
        query: List[str],
        gaps: pd.DataFrame,
        n_lines: int = 100000,
        min_depth: int = 0,
        **kwargs
    ) -> float:
        
        print(gaps)

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

        query_coverage = self.estimate(
            depths,
            **kwargs
        )

        return query_coverage

    def read(
        self,
        vcf: File,
        query: List[str],
        gaps: pd.DataFrame,
        n_lines: int = 100000
    ) -> ArrayLike:
        
        # Load VCF and downsample

        logging.debug(
            f"Downsampling {str(vcf)} to {n_lines} lines..."
        )
                
        vcf_df = self.downsample_vcf(
            vcf = vcf,
            query = query,
            gaps = gaps,
            n_lines = n_lines
        )

        logging.debug(
            f"The downsampled VCF has {len(vcf_df)} variants."
        )
        
        # Exctract depth of coverage at each position
        self.depths = vcf_df[7].apply(
            partial(
                get_info_value,
                tag = "DP",
                dtype = int
            )
        ).values

        logging.debug(
            f"Got {len(self.depths)} valid depths of coverage."
        )

        return self.depths
    
    def estimate(
        self,
        depths: ArrayLike,
        **kwargs
    ) -> float:
        
        return np.median(depths)
    
    def downsample_vcf(
        self,
        vcf: File,
        query: List[str],
        gaps: pd.DataFrame,
        n_lines: int = 100000,
    ) -> pd.DataFrame:
        
        gaps = gaps.reset_index()

        if not len(gaps):
            raise ValueError(
                f"Found no gaps. Cannot estimate the query depth of coverage."
            )
        
        # Create a string for the bcftools view region option
        region = ",".join(
            [ 
                x.chrom + ":" + str(x.start) + "-" + str(
                    x.end
                    if x.end != -1
                    else ""
                )
                for _, x in gaps.iterrows()
            ]
        )

        # Pick n_lines at random from the VCF file
        view_cmd = ["bcftools", "view", "-H", str(vcf), "-r", region]
        shuf_cmd = ["shuf", "-n", str(n_lines)]

        logging.debug(
            f"Downsampling {str(vcf)} with the command \"{' '.join(view_cmd)} | \
{' '.join(shuf_cmd)}\"..."
        )

        view = subprocess.Popen(
            view_cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE
        )
        shuf = subprocess.check_output(
            shuf_cmd,
            stdin = view.stdout
        )

        # Read output as a DataFrame
        return pd.read_table(
            io.StringIO(
                shuf.decode("utf-8")
            ), 
            comment="#", 
            sep="\t", 
            header=None
        )
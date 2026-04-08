import pandas as pd
import numpy as np
from typing import Tuple, Union
from cleansweep.typing import File
from warnings import warn
from scipy import stats as sps
from scipy.optimize import minimize
from scipy.special import gammaln
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
        use_mle: bool = False,
        **kwargs
    ) -> Tuple[float, Union[None, sps.rv_discrete]]:

        dist = None

        if use_mle:
            r, p = self._fit_nbinom_mle(depths)
            logging.debug(f"NB MLE estimates: r = {r:.4f}, p = {p:.4f}")
            dist = sps.nbinom(r, p)
            self.r = r
            self.p = p

        return np.median(depths), dist

    def _fit_nbinom_mle(
        self,
        depths: ArrayLike
    ) -> Tuple[float, float]:
        """Fit Negative Binomial(r, p) parameters by maximum likelihood.

        Uses a reparametrisation (log r, logit p) so that L-BFGS-B optimises
        over an unconstrained space, guaranteeing r > 0 and 0 < p < 1
        regardless of the data variance.
        """
        depths = np.asarray(depths, dtype=float)

        def neg_log_likelihood(params: np.ndarray) -> float:
            r = np.exp(params[0])
            p = 1.0 / (1.0 + np.exp(-params[1]))
            ll = (
                gammaln(depths + r) - gammaln(r)
                + r * np.log(p) + depths * np.log1p(-p)
            ).sum()
            return -ll

        # Initialise from MoM, clamped to a valid range
        mean = float(np.mean(depths))
        var  = float(np.var(depths))
        r0   = max(mean ** 2 / max(var - mean, mean * 0.01), 1e-3)
        p0   = np.clip(mean / max(var, mean + 1e-6), 1e-6, 1.0 - 1e-6)

        result = minimize(
            neg_log_likelihood,
            x0 = [np.log(r0), np.log(p0 / (1.0 - p0))],
            method = "L-BFGS-B"
        )

        if not result.success:
            logging.warning(
                f"NB MLE optimisation did not fully converge: {result.message}. "
                "Using best iterate found."
            )

        r = float(np.exp(result.x[0]))
        p = float(1.0 / (1.0 + np.exp(-result.x[1])))
        return r, p

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
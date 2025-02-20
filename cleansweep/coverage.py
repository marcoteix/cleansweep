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

class CoverageFilter:

    def __init__(self, random_state: int = 23):
        self.random_state = random_state
    
    def fit(self, vcf: pd.DataFrame, query_coverage: float, p_threshold: float = .01, **kwargs) -> pd.DataFrame:
        """Fits a Gaussian Mixture Model with two components to the total_depth of coverage of all 
        variants. Excludes variants assigned to the component of highest mean. The scale parameter
        allows for an adjustment of the FDR by altering the threshold of assignment.

        Args:
            vcf (pd.DataFrame): DataFrame generated from a VCF object. Must have a `total_depth` column
                or a TD tag in the `info` column.
            p_threshold (float, optional): Minimum CDF of the estimated total_depth of coverage
                distribution from correctly called variants for a variant to be included. Controls 
                the FDR. The default is 0.01.

        Returns:
            pd.DataFrame: VCF DataFrame with a `coverage_filter` column appended. Excluded 
                variants have the `coverage_filter` value set to `HighCov`; kept variants have it
                set to `PASS`.
        """

        gm = Pipeline([
            ("scaler", StandardScaler()),
            ("gmm", GaussianMixture(n_components=2, random_state=self.random_state, **kwargs))
        ])
        vcf = self.add_total_depth(vcf)
        gm.fit(vcf[["total_depth"]].values)
        preds = pd.Series(gm.predict(vcf[["total_depth"]].values), index=vcf.index)

        # For each component, set it as "to keep" if the probability of observing its mean 
        # or a smaller value is less than p_threshold, assuming that the coverage of the
        # query follows a Poisson distribution
        means = self.get_means(model=gm)
        distributions = [self.get_distribution(gmm=gm.named_steps["gmm"], include_grp=i) for i in range(2)]
        distributions = [x for x in distributions if self.score_distribution(distribution=x, 
            query_coverage=query_coverage, gmm=gm)]

        # Get p-values
        vcf = vcf.assign(coverage_p = vcf.total_depth.apply(
            partial(self.get_p, distributions=distributions, gmm=gm)))
        
        vcf = vcf.assign(coverage_filter=vcf.coverage_p.lt(p_threshold) \
            .replace({True: "HighCov", False: "PASS"}))

        if not gm.named_steps["gmm"].converged_:
            warn(f"The coverage GMM did not converge. Try increasing the number of iterations.")
        # Keep the GMM object for diagnosis
        self.coverage_gmm = gm

        return vcf
    
    def score_distribution(self, distribution: rv_continuous, query_coverage: float, gmm: Pipeline) -> float:

        # Scale the query coverage
        query_cov_t = gmm.named_steps["scaler"].transform(np.array([[query_coverage]]))
        # How extreme is the expected coverage in the distribution?
        p_val = distribution.cdf(query_cov_t[0,0])

        return p_val>.1
    
    def get_p(self, value: Union[int, float, Collection], distributions: Collection[rv_continuous], gmm: Pipeline):
        
        # Transform value
        value_t = gmm.named_steps["scaler"].transform(np.array([[value]]))

        p_vals = [1-distribution.cdf(value_t[0,0]) for distribution in distributions]
        if not len(p_vals): return 0.0
        else: return max(p_vals)
    
    def get_distribution(self, gmm: GaussianMixture, include_grp: int) -> Any:

        if gmm.means_.shape[1] > 1:
            means = gmm.means_[include_grp]
            cov = gmm.covariances_[include_grp]
            return multivariate_normal(means, cov)
        else:
            means = gmm.means_[include_grp][0]
            sd = gmm.covariances_[include_grp][0][0]

            return norm(means, sd)
    
    def filter(self, vcf: pd.DataFrame) -> pd.DataFrame:

        return vcf[vcf.coverage_filter.eq("PASS")]
    
    def get_means(self, model: Pipeline) -> List[float]:

        return model.named_steps["scaler"] \
            .inverse_transform(model.named_steps["gmm"].means_)
    
    def add_total_depth(self, vcf: pd.DataFrame):

        if not hasattr(vcf, "total_depth"):
            return vcf.assign(total_depth=vcf["info"] \
                .apply(partial(get_info_value, tag="TD", dtype=int)))
        else: return vcf

@dataclass
class CoverageEstimator:

    random_state: int = 23

    def fit(
        self,
        vcf: File,
        query: str,
        gaps: File,
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
        gaps: File,
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
    ) -> Tuple[float, float]:
        
        return np.median(depths), None
    
    def __estimate_deprecated(
        self,
        depths: ArrayLike,
        **kwargs
    ) -> Tuple[float, float]:
        
        # Z-scale depths 
        self.scaler = StandardScaler()
        scaled_depths = self.scaler \
            .fit_transform(
                depths.reshape(-1, 1)
            )
        
        initial_means = self.scaler.transform(
            np.array(
                [1, np.max(depths)]
            ).reshape(-1, 1)
        )

        # Fit 5-component GMM

        logging.debug("Fitting 5-component DPMM...")

        self.gmm = BayesianGaussianMixture(
            n_components = 5,
            random_state = self.random_state,
            weight_concentration_prior = 1e-5,
            max_iter = 10000,
            **kwargs
        )
        self.gmm.fit(scaled_depths)

        # Set the lowest component mean as the estimated depth of coverage
        mean_coverage = np.min(self.gmm.means_)

        # Re-scale means for logging
        rescaled_means = self.scaler \
            .inverse_transform(
                self.gmm.means_
            ).flatten() \
            .astype(str)
        
        logging.debug(
            f"Estimated component means: {', '.join(rescaled_means)}."
        )

        # Set the background coverage as the mean of the component with the most
        # weight, excluding the component assigned to the "true" coverage
        idx_max = np.argmax(
            self.gmm.weight_concentration_[0][
                self.gmm.means_.flatten() != mean_coverage
            ]
        )
        background_coverage = self.gmm.means_ \
            .flatten()[ 
                self.gmm.means_ \
                    .flatten() != mean_coverage
            ][idx_max]
        
        # Re-scale back
        mean_coverage = self.scaler \
            .inverse_transform([[mean_coverage]])[0,0]
        
        background_coverage = self.scaler \
            .inverse_transform([[background_coverage]])[0,0]
        
        logging.debug(f"Estimated mean depth of coverage: {mean_coverage}.")
        
        return mean_coverage, background_coverage

    def downsample_vcf(
        self,
        vcf: File,
        query: str,
        gaps: File,
        n_lines: int = 100000,
    ) -> pd.DataFrame:
        
        # Read gaps file
        gaps = pd.read_csv(
            gaps,
            sep = "\t"
        )

        if not len(gaps):
            raise ValueError(
                f"Found no gaps in {str(gaps)}. Cannot estimate the query depth of coverage."
            )
        
        # Create a string for the bcftools view region option
        region = ",".join(
            [ 
                query + ":" + str(x.start) + "-" + str(x.end)
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
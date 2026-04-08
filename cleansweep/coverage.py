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
from scipy import stats as sps
from scipy.optimize import minimize
from scipy.special import gammaln
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
        
        # Create a string for the bcftools view region option
        region = ",".join(
            [ 
                query + ":" + str(x.start) + "-" + str(
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
        shuf_cmd = [
            "python",
            "-c",
            f"\'import random, sys; file=sys.stdin.readlines(); random.shuffle(file); print(*file[:{n_lines}], sep='\n')\'"
        ]

        logging.debug(
            f"Downsampling {str(vcf)} with the command \"{' '.join(view_cmd)} | \
{' '.join(shuf_cmd)}\"..."
        )

        view = subprocess.Popen(
            view_cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True
        )

        view_out, view_errors = view.communicate()

        if view_errors:
            raise RuntimeError(
                f"Reading {str(vcf)} failed. Command: {' '.join(view_cmd)}. Reason: {view_errors}."
            )
        
        lines = view_out.splitlines()
        random.shuffle(lines)

        lines = lines[:n_lines]

        """
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

        """
        # Read output as a DataFrame
        return pd.read_table(
            io.StringIO("\n".join(lines)), 
            comment="#", 
            sep="\t", 
            header=None
        )
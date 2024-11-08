from functools import partial
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from typing import List, Union, Collection, Callable, Any
from cleansweep.vcf import VCF, get_info_value
from warnings import warn
from scipy.stats import norm, multivariate_normal, poisson, rv_continuous
from numpy.typing import ArrayLike

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
        p_val = distribution.cdf(query_cov_t)

        return p_val>.1
    
    def get_p(self, value: Union[int, float, Collection], distributions: Collection[rv_continuous], gmm: Pipeline):
        
        # Transform value
        value_t = gmm.named_steps["scaler"].transform(np.array([[value]]))

        p_vals = [1-distribution.cdf(value_t) for distribution in distributions]
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
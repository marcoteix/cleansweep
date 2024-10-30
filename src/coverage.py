from functools import partial
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from typing import List, Union, Collection, Callable, Any
from src.vcf import VCF, get_info_value
from warnings import warn
from scipy.stats import norm, multivariate_normal

class CoverageFilter:

    def __init__(self, random_state: int = 23):
        self.random_state = random_state
    
    def fit(self, vcf: pd.DataFrame, p_threshold: float = .1, **kwargs) -> pd.DataFrame:
        """Fits a Gaussian Mixture Model with two components to the total_depth of coverage of all 
        variants. Excludes variants assigned to the component of highest mean. The scale parameter
        allows for an adjustment of the FDR by altering the threshold of assignment.

        Args:
            vcf (pd.DataFrame): DataFrame generated from a VCF object. Must have a `total_depth` column.
            p_threshold (float, optional): Minimum CDF of the estimated total_depth of coverage
                distribution from correctly called variants for a variant to be included. Controls 
                the FDR. The default is 0.1.

        Returns:
            pd.DataFrame: VCF DataFrame with a `coverage_filter` column appended. Excluded 
                variants have the `coverage_filter` value set to `HighCov`; kept variants have it
                set to `PASS`.
        """

        gm = Pipeline([
            ("scaler", StandardScaler()),
            ("gmm", GaussianMixture(n_components=2, random_state=self.random_state, **kwargs))
        ])
        gm.fit(vcf[["total_depth"]].values)
        preds = pd.Series(gm.predict(vcf[["total_depth"]].values), index=vcf.index)

        # Check which component has the highest mean
        means = self.get_means(model=gm)
        include_grp = np.argmin(means)
        # Store the lowest mean as an attribute, to estimate the coverage of the query strain
        self.mean_query_coverage = np.min(means)
        # Get the p-values according to the distribution to include
        vcf = vcf.assign(**{"coverage_p": vcf["total_depth"].apply(
            partial(self.get_p, gm=gm, include_grp=include_grp))})
        fail_flag = "HighCov"

        # Set to PASS all variants with high likelihood for the distribution to include or
        # assigned to that distribution by the GMM
        vcf["coverage_filter"] = (vcf["coverage_p"].le(p_threshold) & preds.ne(include_grp)) \
            .replace({True: fail_flag, False: "PASS"})

        if not gm.named_steps["gmm"].converged_:
            warn(f"The coverage GMM did not converge. Try increasing the number of iterations.")
        # Keep the GMM object for diagnosis
        self.coverage_gmm = gm

        return vcf
    
    def get_p(self, value: Union[int, float, Collection], gm: Pipeline, include_grp: int):

        distribution = self.get_distribution(gm.named_steps["gmm"], include_grp)
        if isinstance(value, (int, float)): value = [value]
        elif isinstance(value, pd.Series): value = value.values
        cdf = distribution.cdf(gm.named_steps["scaler"].transform([value]))
        if not isinstance(cdf, (int, float)): cdf = cdf[0,0]
        return 1-cdf
    
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
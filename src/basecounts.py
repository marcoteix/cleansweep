#%%
from typing import Union
from typing_extensions import Self, Literal
import numpy as np
import pandas as pd 
from dataclasses import dataclass
from sklearn.neighbors import KernelDensity
from numpy.typing import ArrayLike
from scipy.stats import poisson


def add_base_counts(vcf: pd.DataFrame) -> pd.DataFrame:
    """Adds the number of bases supporting the reference and alternate allele to a VCF 
    DataFrame.

    Args:
        vcf (pd.DataFrame): VCF Pandas DataFrame. Must have a \"base_counts\" column.

    Returns:
        pd.DataFrame: VCF Pandas DataFrame with reference and allele basecounts (
            columns \"ref_bc\" and \"alt_bc\", respectively).
    """

    # Order of bases in the VCF BC tag
    bases = {"A": 0, "C": 1, "G": 2, "T": 3}

    return vcf.assign(
        alt_bc=vcf.apply(lambda x: int(x.base_counts.split(",")[bases[x.alt]]), axis=1),
        ref_bc=vcf.apply(lambda x: int(x.base_counts.split(",")[bases[x.ref]]), axis=1)
    )

@dataclass
class BaseCountEstimator:

    kde_bandwith: Union[float, Literal["scott", "silverman"]] = .5

    def __add_base_counts(self, vcf: pd.DataFrame) -> pd.DataFrame:
        return add_base_counts(vcf)

    def __check_vcf(self, vcf: pd.DataFrame):

        for attr in ["base_counts", "filter"]:
            if not hasattr(vcf, attr):
                raise ValueError("The VCF is missing a \"{attr}\" column.")
            
    def __fit_ref_bc_given_alt_allele(self, vcf: pd.DataFrame, kde_kwargs: dict = {}) -> KernelDensity:

        self.ref_kde = KernelDensity(kernel="tophat", 
            bandwidth=self.kde_bandwith, **kde_kwargs)
        samples = vcf[vcf["filter"].eq("PASS")].ref_bc.values.reshape(-1, 1)
        return self.ref_kde.fit(samples)

    def __fit_alt_bc_given_ref_allele(self, vcf: pd.DataFrame, kde_kwargs: dict = {}) -> KernelDensity:

        self.alt_kde = KernelDensity(kernel="tophat", 
            bandwidth=self.kde_bandwith, **kde_kwargs)
        samples = vcf[vcf["filter"].str.contains("Amb")].alt_bc.values.reshape(-1, 1)
        return self.alt_kde.fit(samples)
    
    def fit(self, vcf: pd.DataFrame, kde_kwargs: dict = {}) -> Self:
        """Fits Kernel Density estimators (tophat kernel, bandwith of 0.5) of the conditional
        probabilities of observed reference and alternate allele base counts given that the 
        query contains the alternate and reference alleles, respectively.

        Args:
            vcf (pd.DataFrame): VCF Pandas DataFrame.

        Returns:
            Self: Fitted BaseCountEstimator.
        """
        self.__check_vcf(vcf)
        vcf = self.__add_base_counts(vcf)

        self.__fit_alt_bc_given_ref_allele(vcf, kde_kwargs)
        self.__fit_ref_bc_given_alt_allele(vcf, kde_kwargs)

        return self
    
    def logp_alt_bc_given_ref_allele(self, alt_bc: Union[int, float, pd.Series, ArrayLike]
                                     ) -> Union[float, pd.Series, ArrayLike]:
        """Gives the conditional log-likelihood for an observed alternate allele base count,
        given that the query contains the reference allele. Supports vectorization.

        The object must be fitted first (with `fit()`) before calling this method.

        Args:
            alt_bc (Union[int, float, pd.Series, ArrayLike]): Alternate allele base counts.

        Returns:
            logp (Union[float, pd.Series, ArrayLike]): Conditional log-likelihood for the 
                input base counts.
        """

        return self.__predict_logp(alt_bc, self.alt_kde)

    def logp_ref_bc_given_alt_allele(self, ref_bc: Union[int, float, pd.Series, ArrayLike]
                                     ) -> Union[float, pd.Series, ArrayLike]:
        """Gives the conditional log-likelihood for an observed reference allele base count,
        given that the query contains the alternate allele. Supports vectorization.

        The object must be fitted first (with `fit()`) before calling this method.

        Args:
            ref_bc (Union[int, float, pd.Series, ArrayLike]): Reference allele base counts.

        Returns:
            logp (Union[float, pd.Series, ArrayLike]): Conditional log-likelihood for the 
                input base counts.
        """
        return self.__predict_logp(ref_bc, self.ref_kde)
            
    def __predict_logp(self, base_count: Union[int, float, pd.Series, ArrayLike],
        kde: KernelDensity) -> Union[float, pd.Series, ArrayLike]:
        """Predicts the log-likelihood for a given base count using a fitted KDE. Supports
        vectorization.

        Args:
            base_count (Union[int, float, pd.Series, ArrayLike]): Base counts.
            kde (KernelDensity): Fitted Kernel Density estimator.

        Returns:
            logp (Union[float, pd.Series, ArrayLike]): Log-likelihood for the input base counts.
        """

        if not hasattr(base_count, "__len__"):
            return kde.score_samples(np.array([[base_count]]))
        else:
            # Support vor vectorization
            if isinstance(base_count, pd.Series):
                return pd.Series(
                    kde.score_samples(base_count.values.reshape(-1,1)),
                    index = base_count.index,
                    name="logp_alt_bc_given_ref_allele"
                )
            else:
                return kde.score_samples(base_count.reshape(-1,1))
            
@dataclass
class MAPClassifier:

    reference_ani: float = .998

    def __get_poisson_logp(self, base_counts: Union[int, ArrayLike, pd.Series], 
        expected_coverage: Union[int, float]) -> Union[float, ArrayLike, pd.Series]:

        logp = poisson.logpmf(base_counts, expected_coverage)
        if isinstance(base_counts, pd.Series): 
            return pd.Series(logp, index=base_counts.index, name="poisson_logp")
        else: return logp
    
    def fit(self, vcf: pd.DataFrame, kde_kwargs: dict = {}) -> Self:
        """Fits a maximum a posteriori classifier given observed alternate and reference
        base counts for called variants.

        Args:
            vcf (pd.DataFrame): VCF Pandas DataFrame.
            kde_kwargs (dict, optional): Optional arguments for BaseCountEstimator.

        Returns:
            Self: Fitted MAPClassifier.
        """

        # Fit KDEs for the conditional likelihoods
        self.kde = BaseCountEstimator(**kde_kwargs).fit(vcf)
        return self
    
    def predict(self, vcf: pd.DataFrame, expected_coverage: Union[int, float]) -> pd.Series:
        """Classifies called variants into `PASS` or `FAIL` according to their observed
        alternate and reference base counts. The object must be fitted prior to calling 
        this method. Supports vectorization.

        Args:
            vcf (pd.DataFrame): VCF Pandas DataFrame.
            expected_coverage (Union[int, float]): Expected coverage for the query strain.

        Returns:
            pd.Series: Predicted `PASS` or `FAIL` flags.
        """

        vcf = add_base_counts(vcf)

        # Posterior for the query having the alternate allele
        self.logp_pass = self.kde.logp_ref_bc_given_alt_allele(vcf.ref_bc
            ) + self.__get_poisson_logp(vcf.alt_bc, expected_coverage
            ) + np.log(1-self.reference_ani)
        
        # Posterior for the query having the reference allele
        self.logp_fail = self.kde.logp_alt_bc_given_ref_allele(vcf.alt_bc
            ) + self.__get_poisson_logp(vcf.ref_bc, expected_coverage
            ) + np.log(self.reference_ani)
        
        return self.logp_pass.gt(self.logp_fail).replace({True: "PASS", False:"FAIL"})

    def predict_sample(self, ref_bc: int, alt_bc: int, 
                       expected_coverage: Union[int, float]) -> Literal["PASS", "FAIL"]:

        # Posterior for the query having the alternate allele
        self.logp_pass = self.kde.logp_ref_bc_given_alt_allele(ref_bc
            ) + self.__get_poisson_logp(alt_bc, expected_coverage
            ) + np.log(1-self.reference_ani)
        
        # Posterior for the query having the reference allele
        self.logp_fail = self.kde.logp_alt_bc_given_ref_allele(alt_bc
            ) + self.__get_poisson_logp(ref_bc, expected_coverage
            ) + np.log(self.reference_ani)
        
        return "PASS" if self.logp_pass > self.logp_fail else "FAIL"
# %%

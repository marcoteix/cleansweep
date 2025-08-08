import warnings
from cleansweep_mcmc import MetropolisHastings, SamplingResult
import numpy as np
import pandas as pd 
from typing import Any, Callable, Literal, Union 
from dataclasses import dataclass
from typing_extensions import Self
import scipy.stats as sps
from numpy.typing import ArrayLike
from time import sleep
from tqdm import tqdm
import tqdm.notebook as tqdm_notebook
from threading import Thread

@dataclass
class AlleleDepthFilter:

    query_coverage: float
    samples: int = 10000
    burnin: int = 1000
    chains: int = 5
    dispersion_bias: float = 1.0
    alt_allele_p_proposal_sd: float = 0.1
    dispersion_proposal_sd: float = 0.1
    proposal_p: float = 0.1
    min_acceptance_rate: float = 0.2
    max_acceptance_rate: float = 0.6
    adaptive_step: float = 0.1
    step_size_range: float = 2.0
    block_size: Union[int, float] = 0.05
    threads: int = 1
    random_state: int = 23
    notebook: bool = False

    def __post_init__(self):

        if self.block_size <= 0:
            raise ValueError(f"Block size must be positive. Got {self.block_size}.")

    def fit(self, alt_bc: pd.Series) -> Self:

        # If block size is between 0 and 1, it is a proportion of input sites
        # Convert to a number of sites
        if self.block_size < 1:
            self.block_size = int(self.block_size * len(alt_bc))

        self.__index = alt_bc.index

        # Remove NaNs
        alt_bc = alt_bc.dropna().astype(int)

        self.sampler = MetropolisHastings(
            n_chains = self.chains,
            n_samples = self.samples,
            n_burnin = self.burnin,
            n_cores = self.threads,
            seed = self.random_state,
            dispersion_bias = self.dispersion_bias,
            alt_allele_p_proposal_sd = self.alt_allele_p_proposal_sd,
            dispersion_proposal_sd = self.dispersion_proposal_sd,
            proposal_p = self.proposal_p,
            target_min_acceptance_rate = self.min_acceptance_rate,
            target_max_acceptance_rate = self.max_acceptance_rate,
            adaptive_step_coeff = self.adaptive_step,
            block_size = self.block_size,
            step_size_range = self.step_size_range,
        )

        # Check progress in a concurrent thread
        self.__result = self.sampler.sample(alt_bc, self.query_coverage)

        return self

    def get_sampling_result(self):

        return self.__result
        
    def get_acceptance_rate(self):

        attrs = [
            "alt_allele_p_acceptance_rate", 
            "dispersion_acceptance_rate", 
            "alleles_acceptance_rate"
        ]

        return {
            k: self.__aggregate_result(
                self.__result,
                k,
                function = np.mean
            )
            for k in attrs
        }
    
    def get_distribution_parameters(self):

        params = self.__estimate(self.__result)
        params.dispersion = self.__transform_dispersion(params.dispersion)

        return params
    
    def logpmf(self, depths: Union[pd.Series, ArrayLike, int]) -> float:

        distribution = self.__get_alt_allele_depth_distribution()

        if isinstance(depths, int):
            depths = [depths]

        values = [self.__logpmf(x, distribution) for x in depths] 

        if isinstance(depths, pd.Series):
            values = pd.Series(
                values,
                index = depths.index,
                name = depths.name
            )

        return values 
    
    def cdf(self, depths: Union[pd.Series, ArrayLike, int]) -> float:

        distribution = self.__get_alt_allele_depth_distribution()

        if isinstance(depths, int):
            depths = [depths]

        values = [self.__cdf(x, distribution) for x in depths] 

        if isinstance(depths, pd.Series):
            values = pd.Series(
                values,
                index = depths.index,
                name = depths.name
            )

        return values 
    
    def predict(
        self,
        alt_bc: pd.Series,
        ref_bc: pd.Series,
        power: float = 95.0,
        use_mle: bool = False,
    ) -> pd.Series:
        
        if not isinstance(alt_bc, pd.Series) or not isinstance(ref_bc, pd.Series):
            raise ValueError(
                f"The alternate and reference allele depths should be Pandas \
Series. Got {type(alt_bc)} and {type(alt_bc)}."
            )
        
        # Get log likelihood ratio
        if use_mle:
            probs = self.logpmf(alt_bc) - self.logpmf(ref_bc)
        else:
            probs = self.predict_alleles(bias = 0.5)

        # Convert power to a quantile
        quantile = (1 - power)/2

        # Get CDF
        cdfs = self.cdf(alt_bc)

        probs = pd.Series(probs).where(
            cdfs.between(quantile, 1-quantile),
            pd.NA
        ).rename("p_alt")

        return probs
    
    def get_rhat(self, parameter: Literal["alt_allele_proportion", "dispersion"]) -> float:

        if self.chains < 4:
            warnings.warn("Using less than 4 chains. Rhat may not be reliable.")

        # Whitin-chain variance
        w = np.mean(
            [
                np.var(getattr(x, parameter))
                for x in self.__result.results
            ]
        )

        # Betwee-chain variance
        n_samples = len(self.__result.results[0].dispersion)
        b = n_samples * np.var(
            [
                np.mean(getattr(x, parameter))
                for x in self.__result.results
            ]
        )

        # Marginal posterior variance
        vplus = ((n_samples-1)/n_samples) * w + (b/n_samples)

        return np.sqrt(vplus/w)
    
    def predict_alleles(self, bias: float = 0.5) -> pd.Series:

        # Get mean per allele
        p_alt = np.mean(
            np.vstack(
                [
                    np.array(getattr(x, "alleles")).reshape(self.samples, -1)
                    for x in self.__result.results
                ]
            ),
            axis = 0
        )

        return pd.Series(p_alt, index = self.__index).gt(bias)

    def __aggregate_result(
        self,
        result: SamplingResult,
        parameter: str,
        function: Callable = np.mean
    ) -> Any:
        
        return function(
            [
                getattr(x, parameter)
                for x in result.results
            ]
        )
    
    def __estimate(self, result: SamplingResult) -> pd.Series:

        return pd.Series(
            {
                k: self.__aggregate_result(result, k, function = np.mean)
                for k in [
                    "alt_allele_proportion",
                    "dispersion"
                ]
            }
        )
    
    def __get_alt_allele_depth_distribution(self) -> sps.rv_discrete:
        
        params = self.get_distribution_parameters()

        return sps.nbinom(
            params.dispersion,
            params.dispersion/(params.dispersion + params.alt_allele)
        )

    def __transform_dispersion(self, dispersion: float) -> float:

        return self.query_coverage * 10 ** (dispersion*3 - 1.5)
    
    def __logpmf(self, x: int, distribution: sps.rv_discrete) -> float:

        return distribution.logpmf(x)
    
    def __cdf(self, x: int, distribution: sps.rv_discrete) -> float:

        return distribution.cdf(x)
from cleansweep_mcmc import MetropolisHastings, SamplingResult
import numpy as np
import pandas as pd 
from typing import Any, Callable, Union 
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
    proposal_sd: float = 0.1
    proposal_p: float = 0.1
    block_size: Union[int, float] = 0.05
    threads: int = 1
    random_state: int = 23
    progress: Union[None, float] = None
    notebook: bool = False

    def __post_init__(self):

        if self.block_size <= 0:
            raise ValueError(f"Block size must be positive. Got {self.block_size}.")

    def fit(self, alt_bc: pd.Series) -> Self:

        # If block size is between 0 and 1, it is a proportion of input sites
        # Convert to a number of sites
        if self.block_size < 1:
            self.block_size = int(self.block_size * len(alt_bc))

        # Remove NaNs
        alt_bc = alt_bc.dropna().astype(int)

        self.sampler = MetropolisHastings(
            n_chains = self.chains,
            n_samples = self.samples,
            n_burnin = self.burnin,
            n_cores = self.threads,
            seed = self.random_state,
            dispersion_bias = self.dispersion_bias,
            proposal_sd = self.proposal_sd,
            proposal_p = self.proposal_p,
            block_size = self.block_size,
        )

        # Check progress in a concurrent thread
        if self.progress:
            self.__sample_with_progress(alt_bc)
        else:
            self.__result = self.sampler.sample(alt_bc, self.query_coverage)

        return self

    def get_sampling_result(self):

        return self.__result
        
    def get_acceptance_rate(self):

        return self.__aggregate_result(
            self.__result,
            "acceptance_rate",
            function = np.mean
        )
    
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
        power: float = 95.0
    ) -> pd.Series:
        
        if not isinstance(alt_bc, pd.Series) or not isinstance(ref_bc, pd.Series):
            raise ValueError(
                f"The alternate and reference allele depths should be Pandas \
Series. Got {type(alt_bc)} and {type(alt_bc)}."
            )
        
        # Get log likelihood ratio
        ll_ratio = self.logpmf(alt_bc) - self.logpmf(ref_bc)

        # Convert power to a quantile
        quantile = (1 - power)/2

        # Get CDF
        cdfs = self.cdf(alt_bc)

        ll_ratio = pd.Series(ll_ratio).where(
            cdfs.between(quantile, 1-quantile),
            pd.NA
        ).rename("p_alt")

        return ll_ratio
    
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
    
    def __sample_with_progress(self, alt_bc: pd.Series):

        # Calculate the total number of iterations
        total_iter = self.chains * (self.burnin + self.samples)

        progress_bar_class = (
            tqdm_notebook.tqdm 
            if self.notebook 
            else tqdm
        )

        progress_bar = progress_bar_class(
            desc = "Sampling...",
            total = total_iter
        )

        # Launch sampling in a thread
        result_holder = {}
        def run_sampler():
            result_holder['result'] = self.sampler.sample(alt_bc, self.query_coverage)

        sampler_thread = Thread(target = run_sampler)
        sampler_thread.start()

        # Check progress every self.progress seconds
        while sampler_thread.is_alive():

            progress_bar.n = self.sampler.progress()
            progress_bar.refresh()
            sleep(self.progress)

        progress_bar.n = total_iter
        progress_bar.refresh()
        progress_bar.close()

        sampler_thread.join()
        self.__result = result_holder["result"]
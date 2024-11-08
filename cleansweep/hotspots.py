#%%
from functools import partial
import pandas as pd
import numpy as np
from typing import Union 
from typing_extensions import Self


class HotspotFilter:

    def __init__(self):
        pass

    def fit(self, vcf: pd.DataFrame, window_length: int = 200, max_snps: int = 10) -> pd.DataFrame:
        
        vcf = vcf.assign(n_neighboring_snps=vcf.pos.apply(
            partial(self.n_neighbors, vcf=vcf, window_length=window_length)))
        vcf = vcf.assign(hotspot_filter=vcf.n_neighboring_snps.gt(max_snps).replace(
            {True: "FAIL", False: "PASS"}
        ))

        return vcf
    
    def filter(self, vcf: pd.DataFrame) -> pd.DataFrame:

        if not hasattr(vcf, "hotspot_filter"):
            raise RuntimeError("The VCF DataFrame is missing a \"hotspot_filter\" column.")
        
        return vcf[vcf.hotspot_filter.eq("PASS")]

    def fit_filter(self, vcf: pd.DataFrame, window_length: int = 200, max_snps: int = 10) -> pd.DataFrame:

        vcf = self.fit(vcf, window_length=window_length, max_snps=max_snps)
        return self.filter(vcf)

    def n_neighbors(self, pos: int, vcf: pd.DataFrame, window_length: int) -> int:

        return len(vcf[vcf.pos.ge(pos-window_length/2) & vcf.pos.le(pos+window_length/2)])

# %%

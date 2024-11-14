#%%
from cleansweep.vcf import VCF, get_info_value
from dataclasses import dataclass
from typing import Union
from typing_extensions import Self
from pathlib import Path
import pandas as pd
from functools import partial

FilePath = Union[str, Path]

@dataclass
class InputLoader:

    def load_vcf(self, vcf: FilePath, query: str, **kwargs) -> pd.DataFrame:

        self.__path_exists(vcf)
        return VCF(vcf).read(chrom=query, **kwargs)
    
    def load_coverages(self, coverage: FilePath) -> pd.Series:

        self.__path_exists(coverage)
        return pd.read_table(coverage).set_index("#rname").meandepth

    def __path_exists(self, path: FilePath) -> None:

        if not Path(path).exists():
            raise FileNotFoundError(f"Could not find {str(path)}.")
    
    def load(self, vcf: FilePath, coverage: FilePath, query: str, *, vcf_kwargs: dict = {}) -> Self:

        self.vcf = self.load_vcf(vcf, query, **vcf_kwargs)
        self.coverages = self.load_coverages(coverage)

        return self
    
import pandas as pd
import subprocess
from typing import Union, Collection
import numpy as np 
from dataclasses import dataclass
from pathlib import Path
import logging
from io import StringIO

_VCF_HEADER = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "format", "sample"]


@dataclass
class VCF:

    file: str

    def __post_init__(self):

        if isinstance(self.file, Path): self.file = str(self.file) 

    def read(self, filters: Union[Collection[str], None]=None, include: Union[str, None]=None,
        exclude: Union[str, None]=None) -> pd.DataFrame:
        """Filters a VCF file (compressed or uncompressed) with bcftools. Requires bcftools.

        Args:
            filters (Union[Collection[str], None], optional): Records with these values in the \
FILTER field are included; all others are excluded. If None, includes all values. Defaults to None.
            include (Union[str, None], optional): bcftools expression defining records to include. If \
None, no filters are applied. Defaults to None.
            exclude (Union[str, None], optional): bcftools expression defining records to exclude. If \
None, no filters are applied. Defaults to None.
        """

        filters_cmd = ["-f"] + list(filters) if filters is not None else []
        include_cmd = ["-i"] + list(include) if include is not None else []
        exclude_cmd = ["-e"] + list(exclude) if exclude is not None else []
        command = ["bcftools", "view"] + filters_cmd + include_cmd + exclude_cmd + ["-O", "v", self.file]

        # Run command
        logging.info(f"Running \"{' '.join(command)}\"...")
        rc = subprocess.run(command, stdout=subprocess.PIPE)

        if rc.returncode != 0:
            msg = f"Filtering the VCF file in {self.file} with bcftools failed. Got return \
code {rc.returncode}. Command: \'{' '.join(command)}\'."
            logging.error(msg)
            logging.error("stout dump:")
            logging.error(rc.stdout)
            raise RuntimeError(msg)
        
        # Store filtered VCF as an attribute
        self.vcf = pd.read_csv(StringIO(rc.stdout.decode("utf-8")), sep="\t", comment="#", header=None)  
        self.vcf.columns = _VCF_HEADER[:self.vcf.shape[1]]

        return self.vcf

    def exclude_indels(self, vcf: Union[None, pd.DataFrame]=None) -> pd.DataFrame:

        if vcf is None: 
            inplace = True
            vcf = self.vcf

        vcf = vcf[vcf.ref.str.len().eq(1) & vcf.alt.str.len().eq(1)]
        if inplace: self.vcf = vcf
        return vcf
        
    def include_chrom(self, chroms: Collection[str], 
                      vcf: Union[None, pd.DataFrame]=None) -> pd.DataFrame:

        if vcf is None: 
            inplace = True
            vcf = self.vcf

        vcf = vcf[vcf.chrom.isin(chroms)]
        if inplace: self.vcf = vcf
        return vcf
    
def get_info_value(s:str, tag:str, delim:str = ";", dtype = float):
    return dtype(s.split(tag+"=")[-1].split(delim)[0]) if tag in s else None
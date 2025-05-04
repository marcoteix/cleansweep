from datetime import date
from functools import partial
import sys
import pandas as pd
import subprocess
from typing import Union, Collection
import numpy as np 
from dataclasses import dataclass
from pathlib import Path
import logging
from io import StringIO
from cleansweep.typing import File
from cleansweep.__version__ import __version__

_VCF_HEADER = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "format", "sample"]


@dataclass
class VCF:

    file: File

    def __post_init__(self):

        if isinstance(self.file, Path): self.file = str(self.file) 

    def read(
        self, 
        chrom: Union[str, None], 
        filters: Union[Collection[str], None] = None, 
        include: Union[str, None]=None,
        exclude: Union[str, None]=None,
        add_base_counts: bool = True
    ) -> pd.DataFrame:
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
        self.vcf = pd.read_csv(
            StringIO(rc.stdout.decode("utf-8")), 
            sep="\t", 
            comment="#", 
            header=None
        )  

        self.vcf.columns = [
            x
            for x in rc.stdout \
                .decode("utf-8") \
                .split("\n") 
            if x.startswith("#")
        ][-1].split("\t")

        self.vcf.columns = [
            (
                x.removeprefix("#").lower()
                if x.removeprefix("#").lower() in _VCF_HEADER
                else x.removeprefix("#")
            ) for x in self.vcf.columns
        ]

        # Keep variants in the query
        if chrom:
            self.vcf = self.vcf[self.vcf.chrom.eq(chrom)]

        self.vcf = self.exclude_indels(self.vcf)
        if add_base_counts:
            self.vcf = self.add_info_columns(self.vcf)

        return self.vcf
    
    def get_header(
        self
    ) -> str:
        """Returns the header of this VCF.

        Returns:
            str: Header
        """
        
        command = ["bcftools", "view", str(self.file), "-h"]

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
        
        return rc.stdout.decode("utf-8")

    def exclude_indels(self, vcf: Union[None, pd.DataFrame]=None) -> pd.DataFrame:

        if vcf is None: 
            inplace = True
            vcf = self.vcf
        else:
            inplace = False

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
    
    def add_info_columns(self, vcf: pd.DataFrame) -> pd.DataFrame:
        """Adds the number of bases supporting the reference and alternate allele and the mean mapping
        quality per site as columns to a VCF DataFrame.

        Args:
            vcf (pd.DataFrame): VCF Pandas DataFrame.

        Returns:
            pd.DataFrame: VCF Pandas DataFrame with reference and allele basecounts (
                columns \"ref_bc\" and \"alt_bc\", respectively), and mapping quality (\"mapq\").
        """

        # Order of bases in the VCF BC tag
        bases = {"A": 0, "C": 1, "G": 2, "T": 3}

        if not hasattr(vcf, "base_counts"):
            vcf = vcf.assign(base_counts = vcf["info"] \
                .apply(partial(get_info_value, tag="BC", dtype=str)))
            
        if not hasattr(vcf, "mapq"):
            vcf = vcf.assign(mapq = vcf["info"] \
                .apply(partial(get_info_value, tag="MQ", dtype=int)))
            
        if not hasattr(vcf, "depth"):
            vcf = vcf.assign(depth = vcf["info"] \
                .apply(partial(get_info_value, tag="DP", dtype=int)))
            
        if not hasattr(vcf, "p_alt"):
            vcf = vcf.assign(p_alt = vcf["info"] \
                .apply(
                    lambda x: (
                        10**(-get_info_value(x, tag="CSP", dtype=int))
                        if "CSP=" in x 
                        else pd.NA
                    )
                )
            )
                        
        # If a variant is missing alternate allele information, extract is from the base counts
        vcf.loc[
            vcf.alt.eq("."),
            "alt"
        ] = vcf[
            vcf.alt.eq(".")
        ].apply(
            lambda x: self.__alt_from_base_counts(
                x.base_counts,
                x.ref
            ),
            axis = 1
        )
            
        return vcf.assign(
            alt_bc=vcf.apply(lambda x: int(x.base_counts.split(",")[bases[x.alt]]), axis=1),
            ref_bc=vcf.apply(lambda x: int(x.base_counts.split(",")[bases[x.ref]]), axis=1),
        )
    
    def remove_indels(self, vcf: pd.DataFrame) -> pd.DataFrame:

        return vcf[vcf.alt.ne(".") & vcf.ref.ne(".")]
    
    def __alt_from_base_counts(
        self,
        base_counts: str,
        ref: str
    ) -> str:
        
        # Order of bases in the VCF BC tag
        bases = ["A", "C", "G", "T"]
        
        # Convert base counts to int, setting the base count for the ref allele 
        # to the lowest count
        bc = [
            int(x) if b != ref else -1
            for x, b in zip(
                base_counts.split(","), 
                bases
            ) 
        ]

        # Find the maximum base count (alt allele)
        return bases[np.argmax(bc)]
        
def get_info_value(s:str, tag:str, delim:str = ";", dtype = float):
    return dtype(s.split(tag+"=")[-1].split(delim)[0]) if tag in s else None

def format_vcf_header(
    header: str,
    chrom: Union[None, str] = None,
    ref: Union[None, File, str] = None,
    add_filters: bool = True
) -> str:
    
    lines = []

    for line in header.split("\n"):

        if line.startswith("##fileformat="):
            # Update VCF version
            line = "##fileformat=VCFv4.2"

        elif line.startswith("##fileDate="):
            # Update date
            line = f"##fileDate={date.today().strftime('%Y%m%d')}"

        elif line.startswith("##source="):
            # Update source
            line = f"##source=\"CleanSweep version {__version__}\""

        lines.append(line)

    # Add descriptors for CleanSweep fields
    new_lines = [
        f"##CleanSweepCommand=\"{' '.join(sys.argv)}\""
    ] + (
        [
            "##FILTER=<ID=RefVar,Description=\"Variant also present in the references\">",
            "##FILTER=<ID=FAIL,Description=\"Alternate allele depth does not match the overall depth of coverage\">",
            "##FILTER=<ID=LowAltBC,Description=\"Alternate allele depth is too low\">",
            "##INFO=<ID=PILON,Number=1,Type=String,Description=\"Original Pilon FILTER flag\">",
            "##INFO=<ID=CSP,Number=1,Type=Integer,Description=\"CleanSweep likelihood ratio for a variant being present in the query strain, log transformed\">",
            "##INFO=<ID=RD,Number=1,Type=Integer,Description=\"Reference allele base count\">",
            "##INFO=<ID=AD,Number=1,Type=Integer,Description=\"Main alternate allele base count\">"
        ]
        if add_filters
        else []
    )
        
    if not chrom is None:
        new_lines.append(f"##CleanSweepChrom=\"{chrom}\"")

    if not ref is None:
        new_lines.append(
            f"##CleanSweepReference=\"{str(ref)}\""
        )

    lines = lines[:-2] + new_lines + lines[-2:-1]

    return "\n".join(lines)

def write_vcf(
    vcf: pd.DataFrame,
    file: File,
    header: str,
    chrom: Union[None, str] = None,
):
    
    header = format_vcf_header(
        header,
        chrom = chrom
    )
        
    # Subset the columns in the VCF spec and add fields to INFO 
    fmt_vcf = vcf[
        _VCF_HEADER
    ].rename(
        columns = {
            k: k.upper()
            for k in _VCF_HEADER
        }
    ).drop(
        columns = "FILTER"
    ).assign(
        FILTER = vcf.cleansweep_filter if "cleansweep_filter" in vcf else ".",
        SAMPLE = (
            vcf.cleansweep_filter \
                .eq("PASS") \
                .astype("Int8")
            if "cleansweep_filter" in vcf
            else vcf["sample"]
        ),
        INFO = vcf.apply(
            lambda x: ";".join(
                [
                    x["info"],
                    "PILON=" + x["filter"],
                    "CSP=" + str(
                        int(
                            np.nan_to_num(
                                x["p_alt"]
                                if (
                                    "p_alt" in x and \
                                    not pd.isna(x["p_alt"]) \
                                    and not x["p_alt"] is None
                                )
                                else 1,
                                nan = 1
                            )
                        )
                    ),
                    "RD=" + (
                        str(x.ref_bc)
                        if "ref_bc" in vcf 
                        else "."
                    ),
                    "AD=" + (
                        str(x.alt_bc)
                        if "alt_bc" in vcf 
                        else "."
                    ),                    
                ]
            ),
            axis = 1
        )
    )[ 
        [
            x.upper()
            for x in _VCF_HEADER
        ]
    ]

    # Convert DataFrame to TSV
    vcf_str = "\n".join(
        [
            "\t".join(x.astype(str)) 
            for x in fmt_vcf.values
        ]
    )

    # Write VCF
    with open(file, "w") as out:

        out.write(
            "\n".join(
                [
                    header,
                    vcf_str
                ]
            )
        )

def write_merged_vcf(
    vcf: pd.DataFrame,
    file: File,
    header: str,
):
    
    header = format_vcf_header(
        header,
        chrom = None,
        add_filters = False
    )
        
    # Subset the columns in the VCF spec and add fields to INFO 
    vcf = vcf.rename(
        columns = {
            k: k.upper()
            for k in _VCF_HEADER
        }
    )

    # Convert DataFrame to TSV
    vcf_str = "\n".join(
        [
            "\t".join(x.astype(str)) 
            for x in vcf.values
        ]
    )

    # Write VCF
    with open(file, "w") as out:

        out.write(
            "\n".join(
                [
                    header,
                    vcf_str
                ]
            )
        )
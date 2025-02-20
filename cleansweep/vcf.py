from datetime import date
from functools import partial
import pandas as pd
import subprocess
from typing import Union, Collection
import numpy as np 
from dataclasses import dataclass
from pathlib import Path
import logging
from io import StringIO
from cleansweep.typing import File

_VCF_HEADER = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "format", "sample"]


@dataclass
class VCF:

    file: str

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
        self.vcf.columns = _VCF_HEADER[:self.vcf.shape[1]]

        # Keep variants in the query
        if chrom:
            self.vcf = self.vcf[self.vcf.chrom.eq(chrom)]

        self.vcf = self.exclude_indels(self.vcf)
        if add_base_counts:
            self.vcf = self.add_info_columns(self.vcf)

        return self.vcf

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

def write_vcf(
    vcf: pd.DataFrame,
    file: File,
    chrom: str,
    ref: File,
    version: str
):
    
    header = f"""##fileformat=VCFv4.2
##fileDate={date.today().strftime("%Y%m%d")}
##source="CleanSweep version {version}"
##reference=file:{ref}
##contig=<ID={chrom}>
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=LowAltBC,Description="Low alternate allele depth">
##FILTER=<ID=RefVar,Description="Variant explained by variation between references for the background and query strains">
##FILTER=<ID=LowCov,Description="Low depth of coverage">
##FILTER=<ID=FAIL,Description="Variant not present in the query strain">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Valid read depth; some reads may have been filtered">
##INFO=<ID=TD,Number=1,Type=Integer,Description="Total read depth including bad pairs">
##INFO=<ID=PC,Number=1,Type=Integer,Description="Physical coverage of valid inserts across locus">
##INFO=<ID=BQ,Number=1,Type=Integer,Description="Mean base quality at locus">
##INFO=<ID=MQ,Number=1,Type=Integer,Description="Mean read mapping quality at locus">
##INFO=<ID=QD,Number=1,Type=Integer,Description="Variant confidence/quality by depth">
##INFO=<ID=BC,Number=4,Type=Integer,Description="Count of As, Cs, Gs, Ts at locus">
##INFO=<ID=QP,Number=4,Type=Integer,Description="Percentage of As, Cs, Gs, Ts weighted by Q & MQ at locus">
##INFO=<ID=IC,Number=1,Type=Integer,Description="Number of reads with insertion here">
##INFO=<ID=DC,Number=1,Type=Integer,Description="Number of reads with deletion here">
##INFO=<ID=XC,Number=1,Type=Integer,Description="Number of reads clipped here">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=AF,Number=A,Type=Float,Description="Fraction of evidence in support of alternate allele(s)">
##INFO=<ID=IMPRECISE,Number=0,Type=Flag,Description="Imprecise change from local reassembly (ALT contains Ns)">
##INFO=<ID=PILON,Number=1,Type=String,Description="Original Pilon FILTER flag">
##INFO=<ID=CSP,Number=1,Type=Integer,Description="CleanSweep probability of a variant being present in the query strain, -100*log10 transformed">
##INFO=<ID=RD,Number=1,Type=Integer,Description="Reference allele base count">
##INFO=<ID=AD,Number=1,Type=Integer,Description="Main alternate allele base count">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=.,Type=String,Description="Allelic depths for the ref and alt alleles in the order listed">
##FORMAT=<ID=DP,Number=1,Type=String,Description="Approximate read depth; some reads may have been filtered">
##ALT=<ID=DUP,Description="Possible segmental duplication">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE"""
    
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
        INFO = vcf.apply(
            lambda x: ";".join(
                [
                    x["info"],
                    "PILON=" + x["filter"],
                    "CSP=" + str(
                        np.minimum(
                            int(
                                -100*np.log10(
                                    x["p_alt"] + 1e-100 
                                    if (
                                        "p_alt" in x and \
                                        not pd.isna(x["p_alt"]) 
                                    )
                                    else 1e-100
                                )
                            ),
                            100000
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
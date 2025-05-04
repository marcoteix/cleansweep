#%%
import argparse
import numpy as np
import pandas as pd 
from cleansweep.typing import File, Directory
from typing import List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import subprocess
from cleansweep.vcf import VCF, _VCF_HEADER
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

@dataclass
class Collection:

    vcfs: List[File]
    output: File
    tmp_dir: Directory
    filters: Union[str, None] = None

    def __post_init__(self):

        # Check if the VCFs exist
        for vcf in self.vcfs:

            if not Path(vcf).exists():
                raise FileNotFoundError(
                    f"VCF {str(vcf)} not found."
                )
            
        # Create tmp directory
        self.tmp_dir = Path(self.tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True)

        # Type check filters
        if not self.filters is None:
            if not isinstance(
                self.filters,
                str
            ):
                raise ValueError(
                    f"Filters must be a string, but got {type(self.filters)}."
                )
    
    def prepare_vcfs(
        self,
        vcfs: List[File],
        output_directory: Directory,
        filters: Union[str, None] = None,
        include: str = "PASS,."
    ):
        
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True)

        # Holds the paths to the filtered VCFs
        gzvcfs = {}
        
        for vcf in vcfs:

            # Extract filename
            filename = Path(vcf).name \
                .removesuffix(".gz") \
                .removesuffix(".vcf") \
                .removesuffix(".bcf")

            gzvcf = output_directory.joinpath(
                filename + ".vcf.gz"
            )

            # Convert to gzipped, filter, and index
            command = [
                "bcftools",
                "view",
            ] + (
                ["-f", filters]
                if not filters is None
                else []
            ) + [
                "-i", include,
                "-o", str(gzvcf),
                "-O", "z",
                "--write-index",
                str(vcf)
            ]

            logging.debug(f"Running command \"" + " ".join(command) + "\"...")

            rc = subprocess.run(command)

            self.__raise_run_error(
                f"Filtering VCF {str(vcf)} failed.",
                command,
                rc
            )

            gzvcfs[filename] = gzvcf

    def merge_vcfs(
        self,
        vcfs: List[File],
        output: File
    ):
        
        command = [
            "bcftools",
            "merge",
            "-o", str(output),
            "-O", "z",
            "--force-samples",
            "--missing-to-ref"
        ] + [
            str(x)
            for x in vcfs
        ]

        rc = subprocess.run(command)

        self.__raise_run_error(
            "Merging VCFs failed.",
            command,
            rc
        )

    def add_sample_names_to_vcf(
        self,
        vcf: File,
        names: List[str],
        tmp_dir: Directory,
        output: File
    ):
        
        # Add sample names with bcftools reheader
        
        sample_names_txt = Path(tmp_dir) \
            .joinpath("sample_names.txt")
        
        with open(
            sample_names_txt, "w"
        ) as file:
            
            file.write(
                "\n".join(names)
            )
        
        command = [
            "bcftools",
            "reheader",
            "-s", str(sample_names_txt),
            "-o", str(output),
            str(vcf)
        ]

        rc = subprocess.run(command)

        self.__raise_run_error(
            "Adding sample names to merged VCF failed.",
            command,
            rc
        )

    def merged_vcf_consensus_filter(
        self,
        vcf: File,
        min_snp_distance: int = 1000
    ) -> pd.DataFrame:
        
        # Read VCF
        vcf_df = VCF(vcf).read(
            chrom = None,
            add_base_counts = False
        )

        # Subset genotype columns
        genotype = vcf_df.set_index(
            [
                "chrom",
                "pos"
            ]
        )[vcf_df.columns.difference(_VCF_HEADER)]

        # Compute SNP matrix with all sites
        full_snp_matrix = self.snp_matrix(genotype)

        # Get core SNPs (>1 sample) and compute SNP matrix
        core_snps, is_core = self.core_snps(genotype)
        core_snp_matrix = self.snp_matrix(core_snps)

        # Check Spearman correlation between the full and core SNP distances
        # for each sample. If they are not correlated, use the core SNPs
        for (
            (full_sample_name, full_distances),
            (core_sample_name, core_distances) 
        ) in zip(
            full_snp_matrix.iterrows(),
            core_snp_matrix.iterrows()
        ):

            spearman = spearmanr(
                full_distances,
                core_distances
            )

            if (
                #spearman.pvalue > 0.05 and \
                full_distances.mean() > min_snp_distance
            ):
                                
                # Replace genotype with core
                genotype = genotype.assign(
                    **{
                        full_sample_name: genotype[full_sample_name] \
                            .where(
                                is_core,
                                genotype.mode(axis=1)[0]
                            )
                    }
                )

        return genotype

    def snp_distance(
        self,
        sample1: pd.Series,
        sample2: pd.Series
    ) -> int:

        return int((sample1 != sample2).sum())

    def snp_matrix(
        self,
        genotype: pd.DataFrame
    ) -> pd.DataFrame:
        
        snp_matrix = squareform(
            pdist(
                genotype.transpose(),
                metric = self.snp_distance
            )
        )

        return pd.DataFrame(
            snp_matrix,
            columns = genotype.columns,
            index = genotype.columns
        )

    def core_snps(
        self,
        genotype: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:

        # Number of occurrences across all samples
        n_samples = genotype.sum(axis=1)

        core = n_samples.gt(1) & n_samples.lt(genotype.shape[1]-1)

        return genotype[core], core

    def __raise_run_error(
        self,
        message: str,
        command: List[str],
        rc
    ):
        
        if rc.returncode != 0:
            message = message + f" Got return code {rc.returncode}. Command: \'{' '.join(command)}\'."
            logging.error(message)
            logging.error("stout dump:")
            logging.error(rc.stdout)
            raise RuntimeError(message)
        else:
            logging.debug("Command successful!")
        

# %%

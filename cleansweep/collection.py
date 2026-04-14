#%%
import shutil
import numpy as np
import pandas as pd 
from cleansweep.typing import File, Directory
from typing import List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import subprocess
from cleansweep.vcf import VCF, _VCF_HEADER, write_merged_vcf
from scipy.spatial.distance import pdist, squareform

@dataclass
class Collection:

    vcfs: List[File]
    output: File
    tmp_dir: Directory
    alpha: float = 10.0
    min_coverage: int = 10

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

        if self.alpha <= 0:
            raise ValueError(
                f"Alpha must be greater than 0. Got {self.alpha}."
            )
            
    def merge(self):

        gzvcfs = self.prepare_vcfs(
            vcfs = self.vcfs,
            output_directory = self.tmp_dir,
            min_coverage = self.min_coverage
        )

        self.merge_vcfs(
            vcfs = list(gzvcfs.values()),
            output = self.tmp_dir.joinpath("merged.vcf")
        )

        self.add_sample_names_to_vcf(
            vcf = self.tmp_dir.joinpath("merged.vcf"),
            names = list(gzvcfs.keys()),
            tmp_dir = self.tmp_dir,
            output = self.tmp_dir.joinpath("merged.named.vcf")
        )

        vcf = self.merged_vcf_consensus_filter(
            vcf = self.tmp_dir.joinpath("merged.named.vcf"),
            alpha = self.alpha
        )

        write_merged_vcf(
            vcf = vcf,
            file = self.output,
            header = VCF(
                self.tmp_dir.joinpath("merged.named.vcf")
            ).get_header()
        )

        #shutil.rmtree(self.tmp_dir)
    
    def prepare_vcfs(
        self,
        vcfs: List[File],
        output_directory: Directory,
        filters: Union[None, str] = None,
        min_coverage: int = 10
    ) -> dict:
        
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
                "-i", f"INFO/DP>={min_coverage}",
                "-o", str(gzvcf),
                "-O", "z",
                "--write-index",
                str(vcf)
            ]

            print(f"Running command \"" + " ".join(command) + "\"...")

            rc = subprocess.run(command, capture_output=True)

            self.__raise_run_error(
                f"Filtering VCF {str(vcf)} failed.",
                command,
                rc
            )

            gzvcfs[filename] = gzvcf

        return gzvcfs

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
            #"--missing-to-ref"
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
        alpha: float = 10.0
    ) -> pd.DataFrame:
        
        # Read VCF
        vcf_df = VCF(vcf).read(
            chrom = None,
            add_base_counts = False
        )

        # Get genome length
        genome_length = self.genome_lengths_from_vcf(
            vcf,
            vcf_df.chrom.unique()
        )

        # Subset genotype columns
        genotype = vcf_df.set_index(
            [
                "chrom",
                "pos"
            ]
        )[vcf_df.columns.difference(_VCF_HEADER)] \
        .astype(str)

        # Compute pairwise SNP matrix and convert to ANI
        full_snp_matrix = self.snp_matrix(genotype)
        ani_matrix = 1.0 - full_snp_matrix / genome_length

        # Compute median and IQR of all pairwise ANIs (upper triangle)
        n = len(ani_matrix)
        pairwise_anis = ani_matrix.values[np.triu_indices(n, k=1)]

        if len(pairwise_anis) == 0:
            # Single sample — no filtering possible
            vcf_df = vcf_df.assign(pos=vcf_df.pos.astype("Int64"))
            return vcf_df

        ani_median = float(np.median(pairwise_anis))
        ani_iqr = float(np.percentile(pairwise_anis, 75) - np.percentile(pairwise_anis, 25))
        threshold = ani_median - ani_iqr * alpha

        print(
            f"Pairwise ANI summary: median={ani_median:.6f}, IQR={ani_iqr:.6f}, "
            f"threshold (median - {alpha}*IQR)={threshold:.6f}"
        )

        # Get core SNPs once — reused for every sample that triggers filtering
        _, is_core = self.core_snps(genotype)

        for sample_name in ani_matrix.index:

            # Highest ANI this sample shares with any other sample
            max_ani = float(ani_matrix.loc[sample_name].drop(sample_name).max())

            if max_ani < threshold:

                print(
                    f"Sample {sample_name} has a maximum ANI of {max_ani:.6f} to any other "
                    f"sample, below the threshold of {threshold:.6f} "
                    f"(median={ani_median:.6f}, IQR={ani_iqr:.6f}, alpha={alpha}). "
                    f"Removing non-core SNPs."
                )

                # Replace non-core genotypes with per-site consensus
                genotype = genotype.assign(
                    **{
                        sample_name: genotype[sample_name] \
                            .where(
                                is_core,
                                genotype \
                                    .astype(str) \
                                    .replace(".", pd.NA) \
                                    .mode(
                                        axis = 1,
                                        dropna = True
                                    )[0] \
                                    .astype(str) \
                                    .fillna(".")
                            )
                    }
                )

        vcf_df = genotype.join(
            vcf_df[
                vcf_df.columns.intersection(_VCF_HEADER)
            ].set_index(
                ["chrom", "pos"]
            )
        ).reset_index()[vcf_df.columns]

        vcf_df = vcf_df.assign(
            pos = vcf_df.pos.astype("Int64")
        )

        return vcf_df
    
    def genome_lengths_from_vcf(
        self,
        vcf: File,
        chroms: List[str]
    ) -> int:
        
        with open(vcf) as file:
            content = file.read()

        return sum(
            [
                int(
                    content.split(
                        f"##contig=<ID={chrom},length="
                    )[-1].split(">")[0]
                ) for chrom in chroms
            ]
        )

    def snp_distance(
        self,
        sample1: pd.Series,
        sample2: pd.Series
    ) -> int:

        return int(
            (
                np.logical_and(
                    np.logical_and(
                        sample2 != ".",
                        sample1 != "."
                    ),
                    sample2 != sample1
                )
            ).sum()
        )

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
        n_samples = genotype.eq("1").sum(
            axis = 1,
            numeric_only = True
        )

        # Number of samples with genotype information
        n_pass = genotype.ne(".").sum(axis=1)

        core = (
            n_pass.eq(1) | \
            n_samples.eq(n_pass) | \
            n_samples.eq(0) | \
            (
                n_samples.gt(1) & \
                n_samples.lt(n_pass-1)
            )
        )
        
        return genotype[core], core

    def __raise_run_error(
        self,
        message: str,
        command: List[str],
        rc
    ):
        
        if rc.returncode != 0:
            message = message + f" Got return code {rc.returncode}. Command: \'{' '.join(command)}\'."
            print(message)
            print("stout dump:")
            print(rc.stdout)
            raise RuntimeError(message)
        else:
            print("Command successful!")


# %%

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
from cleansweep.vcf import VCF, _VCF_HEADER, IUPAC_CODES, write_merged_vcf, remove_vcf_header_samples
from scipy.spatial.distance import pdist, squareform

@dataclass
class Collection:

    vcfs: List[File]
    output: File
    tmp_dir: Directory
    alpha: float = 10.0
    min_coverage: int = 10
    exclude: bool = False
    exclude_log: Union[File, None] = None

    def __post_init__(self):

        # Check if the VCFs exist
        for vcf in self.vcfs:

            if not Path(vcf).exists():
                raise FileNotFoundError(f"VCF {str(vcf)} not found.")

        # Create tmp directory
        self.tmp_dir = Path(self.tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True)

        if self.alpha <= 0:
            raise ValueError(f"Alpha must be greater than 0. Got {self.alpha}.")

        if self.exclude_log is not None and not self.exclude:
            raise ValueError(f"--exclude-log ({self.exclude_log}) was given without --exclude.")

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

        vcf, excluded_samples = self.merged_vcf_consensus_filter(
            vcf = self.tmp_dir.joinpath("merged.named.vcf"),
            alpha = self.alpha,
            exclude = self.exclude
        )

        header = VCF(
            self.tmp_dir.joinpath("merged.named.vcf")
        ).get_header()

        if excluded_samples:
            print(
                f"Excluding {len(excluded_samples)} sample(s) from merged VCF: "
                f"{', '.join(excluded_samples)}."
            )
            header = remove_vcf_header_samples(header, excluded_samples)

        write_merged_vcf(
            vcf = vcf,
            file = self.output,
            header = header
        )

        if self.exclude_log is not None:
            with open(self.exclude_log, "w") as file:
                file.write("\n".join(excluded_samples))

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
            filename = Path(vcf).stem

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
        alpha: float = 10.0,
        exclude: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        
        print("Applying consensus filter to merged VCF...")
        
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
        genotype_columns = vcf_df.columns.difference(_VCF_HEADER)

        print(
            f"Found {len(genotype_columns)} samples in merged VCF: "
            f"{', '.join(genotype_columns)}."
        )
        
        genotype = vcf_df.set_index(
            [
                "chrom",
                "pos"
            ]
        )[genotype_columns] \
        .astype(str).drop(
            columns = [
                "Reference", 
                "alt_bc", 
                "base_counts", 
                "depth", 
                "mapq", 
                "p_alt", 
                "ref_bc"
            ],
            errors = "ignore"
        )

        # Compute pairwise SNP matrix and convert to ANI
        full_snp_matrix = self.snp_matrix(genotype)
        ani_matrix = 1.0 - full_snp_matrix / genome_length

        # Get the maximum ANI each sample shares with any other sample
        # (ANI to the most closely related sample)
        max_ani_per_sample = ani_matrix \
            .stack() \
            .to_frame() \
            .reset_index() \
            .rename(
                columns = {
                    "level_0": "sample_1",
                    "level_1": "sample_2",
                    0: "ani"
                }
            ).loc[ 
                lambda x: x.sample_1.ne(x.sample_2)
            ].groupby("sample_1").ani.max()

        # Compute median and IQR
        if len(max_ani_per_sample) < 2:
            # Single sample — no filtering possible
            vcf_df = vcf_df.assign(pos=vcf_df.pos.astype("Int64"))
            return vcf_df, []

        ani_median = float(np.median(max_ani_per_sample))
        ani_iqr = float(np.percentile(max_ani_per_sample, 75) - np.percentile(max_ani_per_sample, 25))
        threshold = ani_median - ani_iqr * alpha

        print(
            f"Maximum ANI summary: median={ani_median:.6f}, IQR={ani_iqr:.6f}, "
            f"threshold (median - {alpha}*IQR)={threshold:.6f}"
        )

        # Samples with a maximum ANI below the threshold
        flagged_samples = [
            sample_name
            for sample_name in ani_matrix.index
            if float(max_ani_per_sample.loc[sample_name]) < threshold
        ]

        if exclude:

            if flagged_samples and len(flagged_samples) == len(ani_matrix.index):
                raise ValueError(
                    f"All {len(ani_matrix.index)} samples were flagged as ANI outliers "
                    f"(below threshold {threshold:.6f}); cannot exclude every sample "
                    "from the merged VCF. Consider increasing --alpha."
                )

            for sample_name in flagged_samples:

                max_ani = float(max_ani_per_sample.loc[sample_name])

                print(
                    f"Sample {sample_name} has a maximum ANI of {max_ani:.6f} to any other "
                    f"sample, below the threshold of {threshold:.6f} "
                    f"(median={ani_median:.6f}, IQR={ani_iqr:.6f}, alpha={alpha}). "
                    f"Excluding sample."
                )

            genotype = genotype.drop(columns=flagged_samples)
            excluded_samples = flagged_samples

        else:

            # Get core SNPs once — reused for every sample that triggers filtering
            consensus, is_core = self.core_snps(genotype)

            print(
                f"Found {is_core.sum()} core SNPs out of {len(is_core)} total SNPs "
                f"({is_core.mean()*100:.2f}%)."
            )

            for sample_name in flagged_samples:

                max_ani = float(max_ani_per_sample.loc[sample_name])

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
                                (
                                    is_core |
                                    genotype[sample_name].eq(consensus) |
                                    genotype[sample_name].eq(".") |
                                    genotype[sample_name].isna()
                                ),
                                consensus
                            )
                    }
                )

            excluded_samples = []

        remaining_columns = [
            c for c in vcf_df.columns if c not in excluded_samples
        ]

        vcf_df = genotype.join(
            vcf_df[
                vcf_df.columns \
                    .intersection(_VCF_HEADER + ["Reference"])
            ].set_index(
                ["chrom", "pos"]
            )
        ).reset_index()[remaining_columns]

        vcf_df = vcf_df.assign(
            pos = vcf_df.pos.astype("Int64")
        )

        return vcf_df, excluded_samples

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
        
        # First, get the most common genotype at each site (the per-site consensus)
        consensus = genotype \
            .astype(str) \
            .replace(".", pd.NA) \
            .mode(
                axis = 1,
                dropna = True
            )[0] \
            .astype(str) \
            .fillna(".")

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

        core = ~(
            n_pass.lt(2) | \
            n_samples.eq(1) | \
            n_samples.eq(n_pass-1)
        )
        
        return consensus, core
    
    def vcf_to_seq(
        vcf: File,
        min_dp: int = 10,
        gt_col: str = "sample",
    ):
        """
        Convert a VCF file to a nucleotide sequence. Handles multi-allelic sites by 
        using IUPAC codes to represent the bases.

        Parameters
        ----------
        vcf : File
            Path to the input VCF file.
        min_dp : int, optional
            Minimum depth of coverage to consider a genotype call valid. Default is 10.
        gt_col : str, optional
            Name of the genotype column in the VCF. Default is "sample".
        
        Returns
        -------
        str
            A string representing the nucleotide sequence, where each position 
            corresponds to a base in the reference genome.
        """
        # Read input VCF
        vcf_df = VCF(vcf).read(None)
        
        # Return an empty string if the VCF is empty
        if vcf_df.empty:
            return ""

        # Allocate an array to hold the sequence
        last_pos = vcf_df.iloc[-1].pos
        seq = np.array(["N"] * last_pos, dtype=str)

        # Iterate through the VCF and fill in the sequence
        # To handle multi-allelic sites, we will use IUPAC codes to represent the bases
        # While parsing the same position, we will collect the bases in a string and 
        # then convert to IUPAC code at the end
        prev_pos, base = 1, ""

        for i, row in vcf_df.iterrows():

            # If the position has changed, update the sequence with the previous base
            # Note that VCF positions are 1-based, so we subtract 1 to get the correct 
            # index in the sequence array
            if row.pos != prev_pos:
                seq[row.pos - 2] = IUPAC_CODES.get("".join(sorted(base)), "N")
                base = ""
                prev_pos = row.pos
            
            # Select between the ref or alt allele
            if row.depth < min_dp or str(row[gt_col]) == ".":
                base = "N"
            elif str(row[gt_col]) == "0":
                base += row.ref
            elif str(row[gt_col]) == "1":
                base += row.alt
            else:
                base = "N"

        # Update the last position in the sequence with the last base
        seq[last_pos - 1] = IUPAC_CODES.get("".join(sorted(base)), "N")

        # Trim the sequence to the last position and return as a string
        return "".join(seq[:last_pos])


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

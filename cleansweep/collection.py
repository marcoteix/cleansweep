#%%
import shutil
import warnings
import numpy as np
import pandas as pd 
from cleansweep.typing import File, Directory
from typing import List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import subprocess
from cleansweep.vcf import VCF, _VCF_HEADER, IUPAC_CODES, write_merged_vcf, remove_vcf_header_samples
from scipy.spatial.distance import pdist, squareform, hamming
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

@dataclass
class Collection:

    vcfs: List[File]
    output: File
    alpha: float = 10.0
    min_coverage: int = 10
    exclude: bool = False
    exclude_log: Union[File, None] = None
    n_threads: Union[None, int] = None

    def __post_init__(self):

        # Check if the VCFs exist
        for vcf in self.vcfs:

            if not Path(vcf).exists():
                raise FileNotFoundError(f"VCF {str(vcf)} not found.")

        if self.alpha <= 0:
            raise ValueError(f"Alpha must be greater than 0. Got {self.alpha}.")

        if self.exclude_log is not None and not self.exclude:
            raise ValueError(f"--exclude-log ({self.exclude_log}) was given without --exclude.")
        
        self.__set_n_threads()
    
    def msa(self):
        """
        Converts a set of VCF files to a multiple sequence alignment (MSA) in FASTA 
        format. Samples that are identified as outliers based on their pairwise 
        similarities will be excluded from the MSA or have their private SNPs removed, 
        depending on the `exclude` parameter.

        If `exclude` is False (default), this method looks for outlier samples: it 
        calculates the maximum average nucleotide identity (ANI) each sample shares with
        any other sample. If a sample's maximum ANI is below the threshold defined by
        the median minus `alpha` times the interquartile range (IQR) of the maximum ANI 
        values, it is considered an outlier. For each outlier sample, any SNPs that are 
        not shared with at least one other sample (i.e., private SNPs) are removed.

        If `exclude` is True, outlier samples are completely excluded from the MSA.
        """

        # Convert each VCF to a sequence. Use multithreading to speed up the process
        # if there are many VCFs.
        with ThreadPool(processes=self.n_threads) as pool:
            sequences = dict(pool.starmap(
                self._vcf_to_seq_item, [(vcf,) for vcf in self.vcfs]
            ))

        # Pad sequences to the same length so downstream operations work correctly
        if sequences:
            max_len = max(len(s) for s in sequences.values())
            sequences = {
                name: seq + "N" * (max_len - len(seq))
                for name, seq in sequences.items()
            }

        # Find outliers
        outliers = self.find_outliers(sequences=sequences, alpha=self.alpha)

        if self.exclude and outliers:
            logging.info(
                f"Excluding {len(outliers)} outlier sample(s) from MSA: "
                f"{', '.join(outliers)}."
            )

            sequences = {
                name: seq
                for name, seq in sequences.items()
                if name not in outliers
            }

            # Write a list of excluded samples to the exclude log file if specified
            if self.exclude_log is not None:
                with open(self.exclude_log, "w") as f:
                    for name in outliers:
                        f.write(f"{name}\n")

        elif outliers:
            logging.info(
                f"Found {len(outliers)} outlier sample(s) in MSA: "
                f"{', '.join(outliers)}. Removing sample-private SNPs..."
            )

            if len(sequences) < 3:
                warnings.warn(
                    "You are trying to use CleanSweep collection with less than 3 "
                    "sequences. This may result in unreliable SNP calls."
                )

            # Generate consensus sequence
            consensus = self.consensus_sequence(sequences=sequences)

            # Remove private SNPs from outliers. Use multithreading to speed up
            # the process if there are many outliers.
            with ThreadPool(processes=self.n_threads) as pool:
                sequences = dict(pool.starmap(
                    self._remove_private_snps_item,
                    [(k, sequences, outliers, consensus) for k in sequences.keys()]
                ))

        # Write MSA to output
        self.write_msa(
            sequences=sequences,
            output_file=self.output
        )
    
    def _vcf_to_seq_item(self, vcf):
        return (Path(vcf).stem, self.vcf_to_seq(vcf=vcf, min_dp=self.min_coverage))

    def _remove_private_snps_item(self, k, sequences, outliers, consensus):
        if k in outliers:
            return (k, self.remove_private_snps(
                target_sequence=sequences[k],
                other_sequences={name: seq for name, seq in sequences.items() if name != k},
                consensus_sequence=consensus
            ))
        return (k, sequences[k])

    def vcf_to_seq(
        self,
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

        # Auto-detect the sample column: it's the 10th column (index 9), after the
        # 9 fixed VCF columns. Named sample columns won't match the "sample" default.
        if gt_col not in vcf_df.columns:
            gt_col = vcf_df.columns[9]

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
                seq[prev_pos - 1] = IUPAC_CODES.get("".join(sorted(base)), "N")
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
    
    def find_outliers(
        self,
        sequences: dict[str, str],
        alpha: float = 3.0
    ):
        """
        Identify outliers in a set of sequences based on their pairwise similarities.
        Outliers are defined as sequences whose minimum similarity to any other sequence
        is below the threshold defined by the median minus `alpha` times the 
        interquartile range (IQR).

        Parameters
        ----------
        sequences : dict[str, str]
            A dictionary where keys are sequence names and values are nucleotide 
            sequences.
        alpha : float, optional
            The multiplier for the interquartile range (IQR) to determine outliers.
            Default is 3.0.

        Returns
        -------
        list[str]
            A list of sequence names identified as outliers.
        """

        if len(sequences) < 2:
            return []

        # Holds the minimum similarity for each sequence between itself
        # and all other sequences
        min_similarities = {}

        for name_1, seq_1 in sequences.items():
            min_similarities[name_1] = np.max(
                [
                    1 - hamming(list(seq_1), list(seq_2))
                    for name_2, seq_2 in sequences.items()
                    if name_1 != name_2
                ]
            )

        # Calculate the median and IQR of the minimum similarities
        median = np.median(list(min_similarities.values()))
        q1 = np.percentile(list(min_similarities.values()), 25)
        q3 = np.percentile(list(min_similarities.values()), 75)
        iqr = q3 - q1

        # Identify outliers based on the IQR method
        outliers = [k for k, v in min_similarities.items()
            if v < (median - alpha * iqr)]
        
        return outliers
    
    def consensus_sequence(
        self,
        sequences: dict[str, str]
    ) -> np.ndarray:
        """
        Generate a consensus sequence from a set of sequences. The consensus at each 
        position is determined by the most common base among the sequences.

        Parameters
        ----------
        sequences : dict[str, str]
            A dictionary where keys are sequence names and values are nucleotide 
            sequences.

        Returns
        -------
        Numpy array
            An array representing the consensus sequence, where each position 
            corresponds to the most common base at that position across all sequences.
        """

        if len(sequences) == 0:
            raise ValueError("No sequences provided for consensus generation.")
        
        def mode(line):
            "A helper function to compute the mode of a list, ignoring '.' and 'N'."
            bases = line[~np.isin(line, [".", "N"])]
            if len(bases) == 0:
                return "N"
            values, counts = np.unique(bases, return_counts=True)
            return values[np.argmax(counts)]
        
        seq_array = np.array([list(seq) for seq in sequences.values()])
        consensus = np.apply_along_axis(mode, 0, seq_array)
        return consensus
    
    def remove_private_snps(
        self,
        target_sequence: str,
        other_sequences: dict[str, str],
        consensus_sequence: np.ndarray
    ) -> str:
        """
        Remove private SNPs from the target sequence by replacing them with the 
        consensus sequence.

        Parameters
        ----------
        target_sequence : str
            The nucleotide sequence from which private SNPs will be removed.
        other_sequences : dict[str, str]
            A dictionary of other sequences to compare against the target sequence.
        consensus_sequence : np.ndarray
            The consensus sequence to use for replacing private SNPs.

        Returns
        -------
        str
            The target sequence with private SNPs replaced by the consensus sequence.
        """

        # Convert sequences to numpy arrays for efficient comparison
        other_seqs = np.array([list(seq) for _, seq in other_sequences.items()])
        target_seq = np.array(list(target_sequence))

        private_snps = np.all(other_seqs != target_seq, axis=0)

        # Replace private SNPs in the target sequence with the consensus
        target_seq[private_snps] = consensus_sequence[private_snps]

        return "".join(target_seq)
    
    def write_msa(
        self,
        sequences: dict[str, str],
        output_file: File
    ):
        """
        Write a multiple sequence alignment (MSA) to a FASTA file.

        Parameters
        ----------
        sequences : dict[str, str]
            A dictionary where keys are sequence names and values are nucleotide 
            sequences.
        output_file : File
            Path to the output FASTA file.
        """

        # Verify that all sequences are of the same length
        lengths = {len(seq) for seq in sequences.values()}
        if len(lengths) > 1:
            raise ValueError(
                "All sequences must be of the same length to write an MSA. "
                f"Found lengths: {lengths}."
            )
        
        with open(output_file, "w") as f:
            for name, seq in sequences.items():
                f.write(f">{name}\n")
                f.write(f"{seq}\n")

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

    def __set_n_threads(self):
        if self.n_threads is None:
            self.n_threads = cpu_count()
        elif self.n_threads < 1:
            raise ValueError(f"n_threads must be greater than 0. Got {self.n_threads}.")
        elif self.n_threads > cpu_count():
            warnings.warn(
                f"n_threads ({self.n_threads}) is greater than the number of available "
                f"CPU cores ({cpu_count()}). Using {cpu_count()} threads instead."
            )
            self.n_threads = cpu_count()


# %%

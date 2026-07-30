"""
Implements a class to call variants from a BAM file using bcftools.

Author: Marco Teixeira
Email: mcarvalh@broadinstitute.org
"""

from dataclasses import dataclass
from cleansweep.align import BwaMem
from typing import Iterable
import numpy as np
from cleansweep.typing import File 
from multiprocessing import cpu_count
from warnings import warn
import subprocess
from pathlib import Path
import logging

@dataclass
class VariantCaller:

    threads: int = 1

    def __post_init__(self):
        
        if not isinstance(self.threads, int):
            raise ValueError(f"Threads should be an integer, got {type(self.threads)} "
                f"({self.threads}).")
        
        if self.threads < 1:
            raise ValueError(f"Got a non-positive number of threads ({self.threads}).")
        
        if self.threads > cpu_count():
            warn(f"Got a number of threads ({self.threads}) greater than the number of "
                f"available CPUs ({cpu_count()}). Using {cpu_count()} threads.")
            self.threads = cpu_count()

    def call(
        self,
        reads: Iterable[File],
        reference: File,
        output: File,
        *,
        method: str = "bcftools",
        strictness: int = 6
    ):
        """
        Aligns paired reads to a reference genome and calls variants using the 
        specified method. Generates a gzip-compressed VCF file as output.

        Parameters
        ----------
        reads : Iterable[File]
            A list of paired-end read files (FASTQ format).
        reference : File
            The reference genome FASTA file.
        output : File
            The output VCF file (gzip compressed).
        method : str, optional
            The variant calling method to use. Supported methods are 'bcftools' and
            'pilon'. Default is 'bcftools'.
        strictness : int, optional
            The strictness parameter for the BWA-MEM alignment. Default is 6.
        """
        aligner = BwaMem(threads = self.threads, strictness = strictness)
        
        directory = Path(output).parent
        
        aligner.align(
            reads = reads,
            reference = reference,
            output = directory / "alignment.bam"
        )

        if method == "bcftools":
            self.call_bcftools(
                bam = directory / "alignment.bam",
                reference = reference,
                output = output
            )
        elif method == "pilon":
            self.call_pilon(
                bam = directory / "alignment.bam",
                reference = reference,
                output = output
            )
        else:
            raise ValueError(f"Unknown method {method}. Supported methods are "
                "'bcftools' and 'pilon'.")

    def call_bcftools(
        self,
        bam: File,
        reference: File,
        output: File
    ):
        """
        Call variants from a BAM file using bcftools.

        Parameters
        ----------
        bam : File
            The input BAM file.
        reference : File
            The reference genome FASTA file.
        output : File
            The output VCF file (gzip compressed).
        """
        
        self.__check_bcftools_installed()
        self.__check_file_exists(bam)
        self.__check_file_exists(reference)

        # Bcftools commands for variant calling
        mpileup_cmd = ["bcftools", "mpileup", "-f", str(reference), "-Ou", str(bam)]
        call_cmd = ["bcftools", "call", "-m", "--ploidy", "1", "-O", "z", "-o", str(output)]

        bcftools_rc = subprocess.run(
            mpileup_cmd, 
            stdout = subprocess.PIPE, 
            stderr = subprocess.PIPE
        )

        if bcftools_rc.returncode:
            msg = f"Generating pileups from {str(bam)} with bcftools failed. Got return "
            f"code {bcftools_rc.returncode}. Command: \'{' '.join(mpileup_cmd)}\'."
            logging.error(msg)
            logging.error("stout dump:")
            logging.error(bcftools_rc.stdout)
            raise RuntimeError(msg)
        
        call_rc = subprocess.run(
            call_cmd,
            input = bcftools_rc.stdout,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE
        )
        
        if call_rc.returncode:
            raise RuntimeError(f"Calling samtools view with command \"{' '.join(call_cmd)}\" "
                f"failed. Error: {call_rc.stderr}.")
        
    def call_pilon(
        self,
        bam: File,
        reference: File,
        output: File
    ):
        """
        Call variants from a BAM file using Pilon.

        Parameters
        ----------
        bam : File
            The input BAM file.
        reference : File
            The reference genome FASTA file.
        output : File
            The output VCF file (gzip compressed).
        """
        
        self.__check_file_exists(bam)
        self.__check_file_exists(reference)
        self.__check_pilon_installed()

        # Calls variants using Pilon
        pilon_cmd = [
            "pilon",
            "--genome", str(reference),
            "--frags", str(bam),
            "--output", str(output).removesuffix(".gz"),
            "--outdir", str(Path(output).parent),
            "--changes", "--vcf",
            "--fix", "bases",
            "--nostrays",
            "--duplicates"
        ]

        # Compress output and index VCF using bcftools
        view_cmd = ["bcftools", "view", "-O", "z", "-o", str(output), 
            str(output).removesuffix(".gz")]

        index_cmd = ["bcftools", "index", str(output)]

        pilon_rc = subprocess.run(pilon_cmd, stdout = subprocess.PIPE, 
            stderr = subprocess.PIPE)

        if pilon_rc.returncode:
            raise RuntimeError(f"Calling Pilon with command \"{' '.join(pilon_cmd)}\" "
                f"failed. Error: {pilon_rc.stderr}.")
    
        view_rc = subprocess.run(view_cmd, stdout = subprocess.PIPE,
            stderr = subprocess.PIPE)

        if view_rc.returncode:
            raise RuntimeError(f"Compressing VCF with command \"{' '.join(view_cmd)}\" "
                f"failed. Error: {view_rc.stderr}.")

        index_rc = subprocess.run(index_cmd, stdout = subprocess.PIPE,
            stderr = subprocess.PIPE)
        
        if index_rc.returncode:
            raise RuntimeError(f"Indexing VCF with command \"{' '.join(index_cmd)}\" "
                f"failed. Error: {index_rc.stderr}.")
        
        # Delete uncompressed VCF
        uncompressed_vcf = Path(output).with_suffix("")
        if uncompressed_vcf.exists():
            uncompressed_vcf.unlink()

    def __check_file_exists(self, file: File):
        """
        Check if a file exists.

        Parameters
        ----------
        file : File
            The file to check.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not Path(file).exists():
            raise FileNotFoundError(f"File {str(file)} does not exist.")
        
    def __check_bcftools_installed(self):
        """
        Check if bcftools is installed.

        Raises
        ------
        RuntimeError
            If bcftools is not installed.
        """
        try:
            rc = subprocess.run(["bcftools", "--version"], check = True, 
                stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("bcftools is not installed or not found in PATH.") from e
        
        if rc.returncode:
            raise RuntimeError("bcftools is not installed or not found in PATH.")
        
    def __check_pilon_installed(self):
        """
        Check if pilon is installed.

        Raises
        ------
        RuntimeError
            If pilon is not installed.
        """
        try:
            rc = subprocess.run(["pilon", "--version"], check = True, 
                stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("pilon is not installed or not found in PATH.") from e

        if rc.returncode:
            raise RuntimeError("pilon is not installed or not found in PATH.")
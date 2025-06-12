import argparse
import logging
from pathlib import Path
from typing import List, Union
from cleansweep.cli.commands import Subcommand
from cleansweep.typing import File, Directory
from cleansweep.align import BwaMem

class AlignCommand(Subcommand):
    """Aligns plate swipe reads to a reference with BWA MEM.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):
        
        parser.add_argument(
            "reads", type=str, nargs="+",
            help = "Plate swipe reads FASTQ file(s)."
        )
        parser.add_argument(
            "--reference", "-r", type=str, required=True,
            help = "Reference FASTA from CleanSweep prepare."
        )
        parser.add_argument(
            "--output", "-o", type=str, required=True, default="alignment.bam",
            help = "Output file name. Defaults to %(default)s."
        )

        aln_grp = parser.add_argument_group("Alignment options")
        aln_grp.add_argument(
            "--strictness", "-s", type=int, default=6,
            help = "Controls how strict the alignment is. The BWA MEM extend, open, \
clip, mismatch, and unpaired penalties will be multiplied by this value. Default \
is %(default)d."
        )
        aln_grp.add_argument(
            "--threads", "-t", type=int, default=1,
            help="Number of threads. Default is %(default)d."
        )
        
    def run(
        self,
        reads: File,
        reference: File,
        output: File = "alignment.bam",
        strictness: float = 4,
        threads: int = 1,
        **kwargs
    ):
                
        bwa_mem = BwaMem(
            threads = threads,
            strictness = strictness
        )

        bwa_mem.align(
            reads = reads,
            reference = reference,
            output = output 
        )
        
        
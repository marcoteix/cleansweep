import argparse
from pathlib import Path
from typing import List, Union
from cleansweep.cli.commands import Subcommand
from cleansweep.nucmer import NucmerAlignment
from cleansweep.typing import File, Directory

class FilterCmd(Subcommand):
    """Prepares a reference for alignment by masking regions in background strains that align
    to the strain of interest.

    Uses nucmer alignment delta files as input and outputs a set of concatenated masked reference
    sequences.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):
        
        parser.add_argument("reference", type=str, help="Reference FASTA file (of the strain of interest).")
        parser.add_argument("query", type=str, nargs="+", help="Query FASTAs for the background strains.")
        parser.add_argument("--output", "-o", type=str, required=True, help="Output concatenated FASTA file.")
        parser.add_argument("--tmp", "-t", type=str, required=False, help="Writes temporary files to this \
directory. If not set, writes to the same directory as the output file.")
        parser.add_argument("--keep-tmp", "-k", action="store_true", help="If set, keeps temporary files.")

        nucmer_grp = parser.add_argument_group("Alignment options")
        nucmer_grp.add_argument("--min-identity", "-mi", type=float, help="Masks alignments with at least this \
identity. Must be between 0 and 1. Default is %(default)f.", default=0.8)
        nucmer_grp.add_argument("--min-length", "-ml", type=int, help="Masks alignments of at least this length. \
Default is %(default)d.", default=150)
          
    def run(
        self,
        reference: File,
        query: List[File],
        output: File,
        tmp: Union[None, Directory],
        keep_tmp: bool,
        min_identity: float,
        min_length: int
    ):
        
        nucmer = NucmerAlignment(
            min_identity = min_identity,
            min_length = min_length
        )

        nucmer.prepare(
            reference = reference,
            queries = query,
            output = output,
            tmp_dir = tmp,
            keep_tmp = keep_tmp
        )
        
        
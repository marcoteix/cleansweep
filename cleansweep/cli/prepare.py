import argparse
import logging
from pathlib import Path
from typing import List, Union
from cleansweep.cli.commands import Subcommand
from cleansweep.nucmer import NucmerAlignment
from cleansweep.typing import File, Directory
from cleansweep.io import StrainGSTParser

class PrepareCmd(Subcommand):
    """Prepares a reference for alignment by masking regions in background strains that align
    to the strain of interest.

    Uses nucmer alignment delta files as input and outputs a set of concatenated masked reference
    sequences.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):
        
        parser.add_argument("reference", type=str, help="Reference FASTA file (of the strain of interest) if \
--file is not provided or the name of the strain of interest if --file is otherwise set.")
        parser.add_argument("--background", "-b", type=str, nargs="+", help="Reference FASTAs for the background \
strains.", required=False)
        parser.add_argument("--output", "-o", type=str, required=True, help="Output directory.")
        parser.add_argument("--file", "-f", type=str, nargs="+", required=False, help="If provided, automatically \
finds the FASTA files for the query and background strains given a set of StrainGST report TSV files with the \
strains detected in this sample. If set, --database must also be specified. The 'reference' argument should be the \
name of the strain of interest, as it appears on the StrainGST output files.")
        parser.add_argument("--database", "-db", type=str, required=False, help="Path to a directory containing \
the StrainGST databases. The FASTA files for the detected strains must all be somewhere in this directory. \
CleanSweep will look for files with a name matching the pattern [strain].fa, with [strain] the name of the \
strains detected by StrainGST in this sample. Make sure there is only one file per strain matching this pattern \
in the directory. Ignored if --file is not provided.")
        parser.add_argument("--tmp", "-t", type=str, required=False, help="Writes temporary files to this \
directory. If not set, writes to the same directory as the output file.")
        parser.add_argument("--keep-tmp", "-k", action="store_true", help="If set, keeps temporary files.")
        parser.add_argument("--verbosity", "-V", type=int, choices = [0, 1, 2, 3, 4], help = "Logging verbosity. \
Ranges from 0 (errors) to 4 (debug). Defaults to %(default)s.", default=1)

        nucmer_grp = parser.add_argument_group("Alignment options")
        nucmer_grp.add_argument("--min-identity", "-mi", type=float, help="Masks alignments with at least this \
identity. Must be between 0 and 1. Default is %(default)f.", default=0.8)
        nucmer_grp.add_argument("--min-length", "-ml", type=int, help="Masks alignments of at least this length. \
Default is %(default)d.", default=150)
        
          
    def run(
        self,
        reference: Union[File, str],
        background: List[File],
        output: File,
        file: Union[List[File], None],
        database: Union[Directory, None],
        tmp: Union[None, Directory],
        keep_tmp: bool,
        min_identity: float,
        min_length: int,
        verbosity: int = 1,
        **kwargs
    ):
        
        outdir = Path(output)
        outdir.mkdir(parents=False, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            filename = outdir.joinpath("cleansweep.prepare.log"),
            filemode = "w",
            encoding = "utf-8",
            level = (4-verbosity) * 10
        )
        
        # Check if a file was provided
        if not file is None:

            if database is None:
                raise ValueError("Provided a --file without setting a --database.")

            straingst_parser = StrainGSTParser()

            # Find FASTAs for each detected strain
            fastas = straingst_parser.get_references(
                query = str(reference),
                straingst_strains = file,
                database_dir = database
            )

            reference = fastas["query"]
            background = fastas["background"]


        nucmer = NucmerAlignment(
            min_identity = min_identity,
            min_length = min_length
        )

        nucmer.prepare(
            reference = reference,
            queries = background,
            output = output,
            tmp_dir = tmp,
            keep_tmp = keep_tmp
        )
        
        
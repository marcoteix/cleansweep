import argparse
import logging
from pathlib import Path
from typing import List, Union
from cleansweep.cli.commands import Subcommand
from cleansweep.collection import Collection
from cleansweep.typing import File, Directory

class CollectionCmd(Subcommand):
    """Merges a set of CleanSweep output VCFs.

    It further filters variants in samples with unreasonable low ANIs with
    other samples, keeping only SNPs occuring in at least two samples. Produces
    a multi-sample VCF with the filtered variants.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):

        io_grp = parser.add_argument_group(
            "Input/Output",
            "Input and output options."
        )

        io_grp.add_argument("input", type=str, nargs="+", help="CleanSweep VCFs to merge.")
        io_grp.add_argument("--output", "-o", type=str, help="Output VCF file.")
        io_grp.add_argument("--tmp-dir", type=str, default="tmp/", 
            help="Temporary directory. Defaults to %(default)s.")

        params_grp = parser.add_argument_group(
            "Filtering options",
            "Options for aditional filtering of variants."
        )

        params_grp.add_argument("--min-ani", "-a", type=float, default=0.995, 
            help="Minimum accepted mean ANI between any sample and every other sample. If \
a sample has a mean ANI with all other samples less than --min-ani, variants occuring in \
no other sample are excluded. Defaults to %(default)s.")
        
    def run(
        self,
        input: List[File],
        output: File,
        tmp_dir: Directory,
        min_ani: float,
        **kwargs
    ):
        
        logging.info(
            f"Merging VCFs {', '.join([str(x) for x in input])} with a minimum \
ANI of {min_ani}. Writing output to {str(output)}..."
        )
        
        Collection(
            vcfs = input,
            output = output,
            tmp_dir = tmp_dir,
            min_ani = min_ani
        ).merge()
        
        logging.info("Done!")
        
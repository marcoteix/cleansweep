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

        params_grp.add_argument("--alpha", "-a", type=float, default=10.0,
            help="Sensitivity of the outlier filter. For each sample, if its highest ANI to \
any other sample is below (median - alpha * IQR) of all pairwise ANIs, variants occurring \
in no other sample are excluded. Larger values are more permissive. Must be > 0. \
Defaults to %(default)s.")
        params_grp.add_argument("--min-coverage", "-c", type=int, default=10,
            help="Minimum coverage needed for a site to be included. Sites with lower \
coverage are represented as N in the multi-sequence alignment. Defaults to %(default)s.")
        
    def run(
        self,
        input: List[File],
        output: File,
        tmp_dir: Directory,
        alpha: float,
        min_coverage: int,
        **kwargs
    ):

        logging.info(
            f"Merging VCFs {', '.join([str(x) for x in input])} "
            f"(alpha={alpha}). Writing output to {str(output)}..."
        )

        Collection(
            vcfs = input,
            output = output,
            tmp_dir = tmp_dir,
            alpha = alpha,
            min_coverage = min_coverage
        ).merge()
        
        logging.info("Done!")
        
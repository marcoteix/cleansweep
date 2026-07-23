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
        io_grp.add_argument("--exclude-log", type=str, default=None,
            help="Path to write the IDs of samples excluded from the merged VCF (one "
            "per line). Only meaningful together with --exclude; raises an error if "
            "given without it. Defaults to no log file.")

        params_grp = parser.add_argument_group(
            "Filtering options",
            "Options for aditional filtering of variants."
        )

        params_grp.add_argument("--alpha", "-a", type=float, default=10.0,
            help="Sensitivity of the outlier filter. For each sample, CleanSweep computes "
            "the highest ANI it shares with any other sample, creating a distribution of "
            "maximum ANIs. If this value is an outlier - below (median - alpha * IQR) of "
            "all maximum ANIs - , variants occurring in no other sample are excluded. Larger "
            "values are more permissive. Must be > 0. Defaults to %(default)s.")
        
        params_grp.add_argument("--min-coverage", "-c", type=int, default=10,
            help="Minimum coverage needed for a site to be included. Sites with lower \
coverage are represented as N in the multi-sequence alignment. Defaults to %(default)s.")

        params_grp.add_argument("--exclude", action="store_true", default=False,
            help="Instead of removing sample-private (non-core) SNPs from samples with an "
            "abnormally low maximum ANI to all other samples, remove those samples "
            "entirely from the merged output VCF. Defaults to %(default)s.")

    def run(
        self,
        input: List[File],
        output: File,
        tmp_dir: Directory,
        alpha: float,
        min_coverage: int,
        exclude: bool,
        exclude_log: Union[File, None],
        **kwargs
    ):

        print(
            f"Merging VCFs {', '.join([str(x) for x in input])} "
            f"(alpha={alpha}, exclude={exclude}). Writing output to {str(output)}..."
        )

        Collection(
            vcfs = input,
            output = output,
            tmp_dir = tmp_dir,
            alpha = alpha,
            min_coverage = min_coverage,
            exclude = exclude,
            exclude_log = exclude_log
        ).merge()
        
        logging.info("Done!")
        
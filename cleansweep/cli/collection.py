import argparse
import logging
from pathlib import Path
from typing import List, Union
from cleansweep.cli.commands import Subcommand
from cleansweep.collection import Collection
from cleansweep.typing import File, Directory

class CollectionCmd(Subcommand):
    """Creates a multisequence alignment (MSA) if FASTA format from a set of 
    CleanSweep VCFs.

    Samples that are identified as outliers based on their pairwise 
    similarities will be excluded from the MSA or have their private SNPs removed, 
    depending on the `--exclude` flag.

    If `--exclude` is not set (default), this method looks for outlier samples: it 
    calculates the maximum average nucleotide identity (ANI) each sample shares with
    any other sample. If a sample's maximum ANI is below the threshold defined by
    the median minus `alpha` times the interquartile range (IQR) of the maximum ANI 
    values, it is considered an outlier. For each outlier sample, any SNPs that are 
    not shared with at least one other sample (i.e., private SNPs) are removed.

    If `--exclude` is set, outlier samples are completely excluded from the MSA.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):

        io_grp = parser.add_argument_group(
            "Input/Output",
            "Input and output options."
        )

        io_grp.add_argument("input", type=str, nargs="+", help="CleanSweep VCFs to merge.")
        io_grp.add_argument("--output", "-o", type=str, help="Output MSA file.")
        io_grp.add_argument("--exclude-log", type=str, default=None,
            help="Path to write the IDs of samples excluded from the MSA (one "
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
            "entirely from the MSA. Defaults to %(default)s.")
        
        parser.add_argument("--n-threads", "-t", type=int, default=1,
            help="Number of threads to use for parallel processing. Defaults to %(default)s.")

    def run(
        self,
        input: List[File],
        output: File,
        alpha: float,
        min_coverage: int,
        exclude: bool,
        exclude_log: Union[File, None],
        n_threads: int,
        **kwargs
    ):

        print(
            f"Creating MSA from VCFs {', '.join([str(x) for x in input])} "
            f"(alpha={alpha}, exclude={exclude}). Writing output to {str(output)}..."
        )

        Collection(
            vcfs=input,
            output=output,
            alpha=alpha,
            min_coverage=min_coverage,
            exclude=exclude,
            exclude_log=exclude_log,
            n_threads=n_threads
        ).msa()
        
        logging.info("Done!")
        
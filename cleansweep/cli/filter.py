import argparse
from pathlib import Path
from typing import Union
from cleansweep.cli.commands import Subcommand
from cleansweep.vcf import VCF
from cleansweep.filter import VCFFilter
from cleansweep.io import InputLoader, FilePath

class FilterCmd(Subcommand):
    """Filters variants in a query strain, called with plate swipe data.

    Reads an output VCF file from Pilon and discards false positive variants
    caused by background reads. Outputs a VCF file with the passing sites.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):
        
        parser.add_argument("input", type=str, help="Input Pilon VCF file.")
        parser.add_argument("query", type=str, help="ID of the query strain, as it \
appears in the alignment files.")
        parser.add_argument("output", type=str, help="Output directory.")
        parser.add_argument("--coverages", "-c", type=str, help="Table of depth of coverages per \
reference strain, obtained with \"samtools coverage\".", required=True)
        parser.add_argument("--seed", "-s", type=int, default=23, help="Random seed.")
        parser.add_argument("--n_chains", "-nc", type=int, default=5, help="Number of MCMC chains. \
Defaults to 5.")
        parser.add_argument("--n_draws", "-nd", type=int, default=10000, help="Number of MCMC sampling \
iterations. Defaults to 10000.")
        parser.add_argument("--n_burnin", "-nb", type=int, default=1000, help="Number of burn-in MCMC \
sampling iterations. Defaults to 1000.")
        parser.add_argument("--bias", "-b", type=float, default=0.5, help="Minimum posterior probability \
of a variant being true needed for a variant to pass the CleanSweep filters. Controls the FDR. Must be a \
value between 0 and 1. Defaults to 0.5.")
        parser.add_argument("--threads", "-t", type=int, default=1, help="Number of threads used in \
MCMC. Defaults to 1.")
        parser.add_argument("--engine", "-e", type=str, default="pymc", choices=["pymc", "numpyro", "nutpie"], 
help="pyMC backend used for NUTS sampling. Default is \"pymc\".")
          
    def run(
        self,
        input: FilePath,
        query: str,
        coverages: FilePath,
        seed: int,
        n_chains: int,
        n_draws: int,
        n_burnin: int,
        bias: float,
        threads: int,
        engine: str,
        output: FilePath
    ):
        
        outdir = Path(output)
        outdir.mkdir(parents=False, exist_ok=True)

        # Read input files
        input_loader = InputLoader().load(vcf=input, coverage=coverages, query=query)

        # Filter
        vcf_filter = VCFFilter(random_state = seed)
        vcf_out = vcf_filter.fit(
            vcf = input_loader.vcf, 
            coverages = input_loader.coverages,
            query_name = query,
            chains = n_chains,
            draws = n_draws,
            burn_in = n_burnin,
            bias = bias,
            threads = threads
        )

        # Save the output VCF DataFrame
        vcf_out.to_csv(outdir.joinpath("cleansweep.variants.tsv"), sep="\t")
        # Save the filter and MCMC results
        vcf_filter.save(outdir.joinpath("cleansweep.filter.pkl"))
        vcf_filter.save_samples(outdir.joinpath("cleansweep.posterior.pkl"))
        
        
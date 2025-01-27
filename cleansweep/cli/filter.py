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
        parser.add_argument("--coverage_min_p", "-p", type=float, default=0.1, help="Controls the filtering \
of high-coverage variants. Greater values lead to more variants being excluded. Must be a value between 0 (do \
not perform coverage-based filtering) and 1. The default is 0.1.")
        parser.add_argument("--min_alt_bc", "-a", type=int, default=10, help="Minimum alternate \
allele base count for a variant to pass the CleanSweep filters. The default is 10.")
        parser.add_argument("--min_ref_bc", "-r", type=int, default=10, help="Variants with fewer than \
this number of reference allele base counts pass the CleanSweep filters automatically.")
        parser.add_argument("--min_ambiguity", "-am", type=float, default=0.0, help="If less than this \
proportion of variants are considered ambiguous by Pilon, skips the base count filter and uses the Pilon \
filters. Defaults to 0 (always apply the base count filter).")
        parser.add_argument("--downsample", "-d", type=float, default=500, help="Number of lines in the \
Pilon output VCF file used to fit the CleanSweep filters. If a float, uses that proportion of lines. Defaults \
to 500.")        
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
        parser.add_argument("--downsampled-vcf", "-v", type=str, help="Downsampled VCF file with a subset of \
the full Pilon output VCF, containing sites in the query.", required=True)
          
    def run(
        self,
        input: FilePath,
        query: str,
        coverages: FilePath,
        downsampled_vcf: FilePath,
        coverage_min_p: float,
        min_alt_bc: int,
        min_ref_bc: int,
        min_ambiguity: float,
        downsample: Union[int, float],
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
        input_loader = InputLoader().load(
            vcf=input, 
            coverage=coverages, 
            query=query
        )

        # Filter
        vcf_filter = VCFFilter(random_state = seed)
        vcf_out = vcf_filter.fit(
            vcf = input_loader.vcf, 
            coverages = input_loader.coverages,
            query_name = query,
            downsampled_vcf = downsampled_vcf,
            coverage_min_p = coverage_min_p,
            min_alt_bc = min_alt_bc,
            min_ref_bc = min_ref_bc,
            min_ambiguity = min_ambiguity,
            downsampling = downsample,
            chains = n_chains,
            draws = n_draws,
            burn_in = n_burnin,
            bias = bias,
            threads = threads,
            engine = engine
        )

        # Save the output VCF DataFrame
        vcf_out.to_csv(outdir.joinpath("cleansweep.variants.tsv"), sep="\t")
        # Save the filter and MCMC results
        vcf_filter.save(outdir.joinpath("cleansweep.filter.pkl"))
        vcf_filter.save_samples(outdir.joinpath("cleansweep.posterior.pkl"))
        
        
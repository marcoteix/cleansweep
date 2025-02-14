import argparse
from pathlib import Path
from typing import Iterable, Union
from cleansweep.cli.commands import Subcommand
from cleansweep.vcf import write_vcf
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
        parser.add_argument("--min_depth", "-dp", type=int, default=5, help="Minimum depth of coverage \
for a site to be considered when filtering SNPs. SNPs at sites with fewer than this number of reads \
in pileups are automatically excluded and these sites are also ignored when estimating the depth of \
coverage for the query strain.")
        parser.add_argument("--min_alt_bc", "-a", type=int, default=10, help="Minimum alternate \
allele base count for a variant to pass the CleanSweep filters. The default is 10.")
        parser.add_argument("--min_ref_bc", "-r", type=int, default=10, help="Variants with fewer than \
this number of reference allele base counts pass the CleanSweep filters automatically.")
        parser.add_argument("--downsample", "-d", type=float, default=500, help="Number of lines in the \
Pilon output VCF file used to fit the CleanSweep filters. If a float, uses that proportion of lines. Defaults \
to 500.") 
        parser.add_argument("--max-overdispersion", "-v", type=float, default=0.55, help="Maximum \
overdispersion for the depth of coverage of the query strain. This value is only used to detect variants \
with low alternate allele base counts not reported by the variant caller. Increasing this overdispersion \
will lead to more variants being called, with lower alternate allele base counts. This increases recall \
but may lead to a decrease in precision. The actual overdispersion estimated by CleanSweep may be greater \
than this value.")         
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
        parser.add_argument("--n-coverage-sites", "-Nc", type=int, help="Number of sites used to estimate the \
query depth of coverage. Defaults to %(default)s.", default=100000)
        parser.add_argument("--nucmer_snps", "-snp", type=str, nargs="+", help="List of SNPs detected by \
nucmer between the reference sequence for the query strain and each of the background reference sequences. \
Can be generated with the show-snps subcommand of nucmer.", required=True)
          
    def run(
        self,
        input: FilePath,
        query: str,
        nucmer_snps: Iterable[FilePath],
        n_coverage_sites: int,
        min_depth: int,
        min_alt_bc: int,
        min_ref_bc: int,
        max_overdispersion: float,
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

        # Set a temporary directory
        tmp_dir = Path(output) \
            .joinpath("tmp")
        tmp_dir.mkdir(
            exist_ok = True,
            parents = True
        )

        # Filter
        vcf_filter = VCFFilter(random_state = seed)
        vcf_out = vcf_filter.fit(
            vcf = input, 
            query = query,
            nucmer_snps = nucmer_snps,
            tmp_dir = tmp_dir,
            n_coverage_sites = n_coverage_sites,
            min_depth = min_depth,
            min_alt_bc = min_alt_bc,
            min_ref_bc = min_ref_bc,
            max_overdispersion = max_overdispersion,
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
        
        
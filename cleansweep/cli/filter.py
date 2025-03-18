import argparse
from pathlib import Path
from typing import Iterable, Union

import joblib
from cleansweep.cli.commands import Subcommand
from cleansweep.vcf import write_vcf
from cleansweep.filter import VCFFilter
from cleansweep.typing import File, Directory
from cleansweep.__version__ import __version__
import logging

class FilterCmd(Subcommand):
    """Filters variants in a query strain, called with plate swipe data.

    Reads an output VCF file from Pilon and discards false positive variants
    caused by background reads. Outputs a VCF file with the passing sites.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):

        io_grp = parser.add_argument_group(
            "Input/Output",
            "Input and output options."
        )
        
        io_grp.add_argument("input", type=str, help="Input Pilon VCF file.")
        io_grp.add_argument("prepare", type=str, help="Output .swp file from CleanSweep prepare \
(cleansweep.prepare.swp).")
        io_grp.add_argument("output", type=str, help="Output directory.")
        io_grp.add_argument("--verbosity", "-V", type=int, choices = [0, 1, 2, 3, 4], help = "Logging verbosity. \
Ranges from 0 (errors) to 4 (debug). Defaults to %(default)s.", default=1)

        params_grp = parser.add_argument_group(
            "Filtering options",
            "Parameters and options for the CleanSweep filter."
        )

        params_grp.add_argument("--min-depth", "-dp", type=int, default=5, help="Minimum depth of coverage \
for a site to be considered when filtering SNPs. SNPs at sites with fewer than this number of reads \
in pileups are automatically excluded and these sites are also ignored when estimating the depth of \
coverage for the query strain.")
        params_grp.add_argument("--min-alt-bc", "-a", type=int, default=10, help="Minimum alternate \
allele base count for a variant to pass the CleanSweep filters. The default is 10.")
        params_grp.add_argument("--min-ref-bc", "-r", type=int, default=10, help="Variants with fewer than \
this number of reference allele base counts pass the CleanSweep filters automatically.")
        params_grp.add_argument("--downsample", "-d", type=float, default=500, help="Number of lines in the \
Pilon output VCF file used to fit the CleanSweep filters. If a float, uses that proportion of lines. Defaults \
to 500.") 
        params_grp.add_argument("--max-overdispersion", "-v", type=float, default=0.55, help="Maximum \
overdispersion for the depth of coverage of the query strain. This value is only used to detect variants \
with low alternate allele base counts not reported by the variant caller. Increasing this overdispersion \
will lead to more variants being called, with lower alternate allele base counts. This increases recall \
but may lead to a decrease in precision. The actual overdispersion estimated by CleanSweep may be greater \
than this value.")
        params_grp.add_argument("--overdispersion-bias", "-ob", type=int, help="Controls the overdispersion \
prior for the query strain. More specifically, it is the value of the alpha and beta parameters of a Beta \
distribution: greater values lead to an overdispersion closer to 0.5. Defaults to %(default)s.", default=500)
        params_grp.add_argument("--n-coverage-sites", "-Nc", type=int, help="Number of sites used to estimate the \
query depth of coverage. Defaults to %(default)s.", default=100000)
        params_grp.add_argument("--seed", "-s", type=int, default=23, help="Random seed.")

        mcmc_grp = parser.add_argument_group(
            "MCMC options",
            "Options for MCMC estimation."
        )

        mcmc_grp.add_argument("--n-chains", "-nc", type=int, default=5, help="Number of MCMC chains. \
Defaults to 5.")
        mcmc_grp.add_argument("--n-draws", "-nd", type=int, default=10000, help="Number of MCMC sampling \
iterations. Defaults to 10000.")
        mcmc_grp.add_argument("--n-burnin", "-nb", type=int, default=1000, help="Number of burn-in MCMC \
sampling iterations. Defaults to 1000.")
        mcmc_grp.add_argument("--threads", "-t", type=int, default=1, help="Number of threads used in \
MCMC. Defaults to 1.")
        mcmc_grp.add_argument("--engine", "-e", type=str, default="pymc", choices=["pymc", "numpyro", "nutpie"], 
help="pyMC backend used for NUTS sampling. Default is \"pymc\".")
        
    def run(
        self,
        input: File,
        prepare: File,
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
        threads: int,
        engine: str,
        verbosity: int,
        output: Directory,
        overdispersion_bias: int,
        **kwargs
    ):
        
        outdir = Path(output)
        outdir.mkdir(parents=False, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            filename = outdir.joinpath("cleansweep.filter.log"),
            filemode = "w",
            encoding = "utf-8",
            level = (4-verbosity) * 10
        )

        # Set a temporary directory
        tmp_dir = Path(output) \
            .joinpath("tmp")
        logging.debug(f"Creating a temporary directory in {str(tmp_dir)}...")
        tmp_dir.mkdir(
            exist_ok = True,
            parents = True
        )

        # Read CleanSweep prepare file
        logging.debug(f"Reading CleanSweep prepare file in {str(prepare)}...")
        prepare_dict = joblib.load(prepare)

        # Filter

        logging.info(f"Filtering {str(input)}, contigs {', '.join(prepare_dict['chrom'])}...")

        vcf_filter = VCFFilter(random_state = seed)
        vcf_out = vcf_filter.fit(
            vcf = input, 
            gaps = prepare_dict["gaps"],
            query = prepare_dict["chrom"][0],
            nucmer_snps = prepare_dict["snps"],
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
            threads = threads,
            engine = engine,
            overdispersion_bias = overdispersion_bias
        )

        # Write the output VCF
        logging.debug(
            f"Writing filtered VCF to {str(outdir.joinpath("cleansweep.variants.vcf"))}..."
        )
        
        write_vcf(
            vcf = vcf_out,
            file = outdir.joinpath("cleansweep.variants.vcf"),
            chrom = prepare_dict['chrom'][0],
            ref = "unknown",
            version = __version__
        )

        # Save the filter and MCMC results
        vcf_filter.save(outdir.joinpath("cleansweep.filter.swp"))
        vcf_filter.save_samples(outdir.joinpath("cleansweep.posterior.swp"))
        
        logging.info("Done!")
        
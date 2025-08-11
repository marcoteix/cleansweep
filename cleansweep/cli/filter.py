import argparse
from pathlib import Path
from typing import Union, Literal
import joblib
from cleansweep.cli.commands import Subcommand
from cleansweep.vcf import write_vcf, write_full_vcf, VCF
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
        io_grp.add_argument("--prefix", "-p", type=str, default="cleansweep", help="Prefix added to output files. \
Defaults to \"%(default)s\".")
        io_grp.add_argument("--variants", action="store_true", help="If set, only writes sites evaluated by \
CleanSweep, ignoring sites with no evidence of an alternate allele. Writes all sites by default.")
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
        params_grp.add_argument("--downsample", "-d", type=float, default=-1, help="Number of lines in the \
Pilon output VCF file used to fit the CleanSweep filters. If a float, uses that proportion of lines. If less \
than zero, uses all lines. Defaults to %(default)s.") 
        params_grp.add_argument("--max-dispersion", "-v", type=float, default=0.1, help="Maximum \
dispersion for the depth of coverage of the query strain. This value is only used to detect variants \
with low alternate allele base counts not reported by the variant caller. Increasing this dispersion \
will lead to more variants being called, with lower alternate allele base counts. This increases recall \
but may lead to a decrease in precision. The actual dispersion estimated by CleanSweep may be greater \
than this value.")
        params_grp.add_argument("--dispersion-bias", "-ob", type=float, help="Controls the dispersion \
prior for the query strain. More specifically, it is the value of the alpha and beta parameters of a Beta \
distribution: greater values lead to an dispersion closer to 0.5. Defaults to %(default)s.", default=1)
        params_grp.add_argument("--n-coverage-sites", "-Nc", type=int, help="Number of sites used to estimate the \
query depth of coverage. Defaults to %(default)s.", default=100000)
        params_grp.add_argument("--use-mle", type=str, choices=["true", "false", "auto"], default="auto",
            help="Whether CleanSweep should use the MLE estimator (\"true\") or the posterior distributions \
(\"false\") to predict alleles. If \"auto\", uses the MLE if the number of variants passed to the MCMC \
(see --downsample) is less than the total number of candidate variants, and the posterior otherwise. Note \
that you may not use the posterior if downsampling is enabled. Defaults to %(default)s.")
        params_grp.add_argument("--seed", "-s", type=int, default=23, help="Random seed.")

        mcmc_grp = parser.add_argument_group(
            "MCMC options",
            "Options for MCMC estimation."
        )

        mcmc_grp.add_argument("--n-chains", "-nc", type=int, default=5, help="Number of MCMC chains. \
Defaults to %(default)s.")
        mcmc_grp.add_argument("--n-draws", "-nd", type=int, default=10000, help="Number of MCMC sampling \
iterations. Defaults to %(default)s.")
        mcmc_grp.add_argument("--n-burnin", "-nb", type=int, default=1000, help="Number of burn-in MCMC \
sampling iterations. Defaults to %(default)s.")
        mcmc_grp.add_argument("--threads", "-t", type=int, default=1, help="Number of threads used in \
MCMC. Defaults to %(default)s.")
        mcmc_grp.add_argument("--alt-allele-p-step-size", "-aps", type=float, default=0.1, 
            help="Standard deviation of the proposal (normal) distribution for the alternate allele \
proportion. Decreasing it should increase the acceptance rate for this parameter. Defaults to %(default)s.")
        mcmc_grp.add_argument("--dispersion-step-size", "-ds", type=float, default=0.1, 
            help="Standard deviation of the proposal (normal) distribution for the dispersion parameter. \
Decreasing it should increase the acceptance rate for this parameter. Defaults to %(default)s.")
        mcmc_grp.add_argument("--allele-step-size", "-as", type=float, default=0.1, 
            help="Probability of changing an allele in each MCMC iteration. Decreasing it should increase \
the acceptance rate for this parameter. Defaults to %(default)s.")
        mcmc_grp.add_argument("--min-acceptance-rate", type=float, default=0.2, 
            help="Target minimum acceptance rate. If the acceptance rate for a parameter is lower than \
this value, the step size is decreased proportionally to the difference, multiplied by --adaptive-step. \
Defaults to %(default)s.")
        mcmc_grp.add_argument("--max-acceptance-rate", type=float, default=0.6, 
            help="Target maximum acceptance rate. If the acceptance rate for a parameter is greater than \
this value, the step size is increased proportionally to the difference, multiplied by --adaptive-step. \
Defaults to %(default)s.")
        mcmc_grp.add_argument("--adaptive-step", type=float, default=0.1, 
            help="Proportional factor to update step sizes (see --min-acceptance-rate and \
--max-acceptance-rate). Defaults to %(default)s.")
        mcmc_grp.add_argument("--block-size", type=float, default=0.05, 
            help="Fraction of alleles updated simultaneously. Increasing this value will make CleanSweep \
filter faster, but the MCMC may not converge. Defaults to %(default)s.")

    def run(
        self,
        input: File,
        prepare: File,
        n_coverage_sites: int,
        min_depth: int,
        min_alt_bc: int,
        min_ref_bc: int,
        max_dispersion: float,
        downsample: Union[int, float],
        seed: int,
        n_chains: int,
        n_draws: int,
        n_burnin: int,
        threads: int,
        verbosity: int,
        output: Directory,
        dispersion_bias: int,
        alt_allele_p_step_size: float,
        dispersion_step_size: float,
        allele_step_size: float,
        min_acceptance_rate: float,
        max_acceptance_rate: float,
        adaptive_step: float,
        block_size: float,
        use_mle: Literal["auto", "true", "false"],
        variants: bool,
        prefix: str,
        **kwargs
    ):
        if use_mle == "auto":
            use_mle = None 
        elif use_mle == "false":
            use_mle = False 
        elif use_mle == "true":
            use_mle = True
        
        outdir = Path(output)
        outdir.mkdir(parents=False, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            filename = outdir.joinpath(prefix + ".filter.log"),
            filemode = "w",
            encoding = "utf-8",
            level = (4-verbosity) * 10
        )

        # Set a temporary directory
        tmp_dir = Path(output) \
            .joinpath(prefix + "_tmp")
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
            max_dispersion = max_dispersion,
            downsampling = downsample,
            chains = n_chains,
            draws = n_draws,
            burn_in = n_burnin,
            threads = threads,
            dispersion_bias = dispersion_bias,
            alt_allele_p_step_size = alt_allele_p_step_size,
            dispersion_step_size = dispersion_step_size,
            allele_step_size = allele_step_size,
            min_acceptance_rate = min_acceptance_rate,
            max_acceptance_rate = max_acceptance_rate,
            adaptive_step = adaptive_step,
            block_size = block_size,
            use_mle = use_mle
        )

        # Write the output VCF
        logging.debug(
            f'Writing filtered VCF to {str(outdir.joinpath(prefix + ".variants.vcf"))}...'
        )
        
        if variants:
            write_vcf(
                vcf = vcf_out,
                file = outdir.joinpath(prefix + ".variants.vcf"),
                header = VCF(str(input)).get_header(),
                chrom = prepare_dict['chrom'],
            )
        else:
            write_full_vcf(
                vcf = vcf_out,
                full_vcf = input,
                file = outdir.joinpath(prefix + ".variants.vcf"),
                header = VCF(str(input)).get_header(),
                chrom = prepare_dict['chrom'],
                min_dp = min_depth
            )
        
        vcf_filter.save(
            outdir.joinpath(prefix + ".filter.swp")
        )

        logging.info("Done!")
        
import argparse
from cleansweep import CleanSweep
from cleansweep.typing import File, Directory
from cleansweep.cli.commands import Subcommand
from cleansweep.filter import VCFFilter
from cleansweep.vcf import write_vcf, write_full_vcf
from pathlib import Path
import joblib
from typing import Union, Literal
import logging
from cleansweep.vcf import VCF

class CallCmd(Subcommand):
    """
    Calls single-nucleotide variants for a target strain reference from plate
    sweep reads. 

    CleanSweep determines the alleles present in a target strain in a plate sweep
    based on allele depths: it searches for allele depths consistent with the 
    depth of coverage along the entire genome of the target strain.

    We use the term 'background strains' to refer to the other strains present in 
    the plate sweep that are not the target strain.

    This subcommand runs the entire CleanSweep pipeline end-to-end. It starts by
    preparing a reference for read alignment and variant calling by:
        1. Aligning every background strain reference to the target strain reference;
        2. Masking regions of the background strain references that align to the
        target strain reference;
        3. Identifying SNPs between the target and background strain references.
        These positions are assumed to contain the reference allele;
        4. Estimating the depth of coverage of the target strain in the plate sweep
        based on the coverage along regions unique to the target strain reference.
    
    Then, it aligns the plate sweep reads to the target strain reference and calls
    variants using Pilon or bcftools. Finally, it filters the variants based on the
    depth of coverage in the target strain. 

    CleanSweep outputs a VCF file with FILTER information indicating whether a 
    variant is truly present in the target strain or not.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):

        # ---------------- IO options ----------------

        io_grp = parser.add_argument_group(
            "Input/Output",
            "Input and output options."
        )

        io_grp.add_argument(
            "--reads1", "-r1", type=str,
            help="Path to the first read file containing plate sweep reads (FASTQ "
            "format, paired or single-end)."
        )

        io_grp.add_argument(
            "--reads2", "-r2", type=str, required=False,
            help="Path to the second read file containing plate sweep reads (FASTQ "
            "format). Leave empty for single-end reads."
        )

        io_grp.add_argument(
            "--reference", "-r", type=str, required=True,
            help="Path to the reference genome of the target strain (FASTA format). " \
            "Alternatively, you can provide a reference name from the StrainGST " \
            "database (e.g., 'Escherichia_coli_K12') if you specify "
            "--straingst-results and --straingst-database."
        )

        io_grp.add_argument(
            "--background", "-b", type=str, nargs="+", required=False,
            help="Path(s) to the reference genome(s) of the background strain(s) "
            "(FASTA format). These are the other strains present in the plate sweep "
            "that are not the target strain. If you provide StrainGST results with "
            "--straingst-results, this option will be ignored."
        )

        io_grp.add_argument(
            "--straingst-results", type=str, required=False,
            help="Optional path to the StrainGST results TSV file for the plate sweep "
            "reads. If provided along with --straingst-database, CleanSweep will "
            "automatically find the reference and background FASTAs. The target strain "
            "will be the one specified in --reference, and all other detected strains "
            "will be used as background strains."
        )

        io_grp.add_argument(
            "--straingst-database", type=str, required=False,
            help="Optional path to the StrainGST database directory. Required if "
            "you provide --straingst-results."
        )

        io_grp.add_argument(
            "--tmp-dir", type=str, required=False,
            help="Path to a temporary directory for intermediate files. If not "
            "provided, a temporary directory will be created in the output directory."
        )

        io_grp.add_argument(
            "--keep-tmp", action="store_true",
            help="If set, keeps the temporary directory and intermediate files. By "
            "default, the temporary directory is deleted after CleanSweep finishes."
        )

        io_grp.add_argument(
            "--output", "-o", type=str, required=True,
            help="Path to the output directory where CleanSweep results will be "
            "saved."
        )

        io_grp.add_argument(
            "--prefix", "-p", type=str, default="cleansweep",
            help="Prefix for the output VCF file. The name of the final VCF will be "
            "<prefix>.vcf. Default is %(default)s."
        )

        # ---------------- Prepare options ----------------

        prepare_grp = parser.add_argument_group(
            "Prepare options",
            "Parameters and options for preparing a reference for read alignment and "
            "variant calling."
        )

        prepare_grp.add_argument(
            "--min-mask-identity", "-mi", type=float, default=0.95,
            help="Minimum mask identity for masking regions of the background strain "
            "references. Regions of the background strain references aligning to the "
            "target reference with at least this identity will be masked. Default is "
            "%(default)s."
        )

        prepare_grp.add_argument(
            "--min-mask-length", "-ml", type=int, default=100,
            help="Minimum mask length for masking regions of the background strain "
            "references. Regions of the background strain references aligning to the "
            "target reference with at least this length will be masked. Default is "
            "%(default)s."
        )

        # ---------------- Variant calling options ----------------

        variant_grp = parser.add_argument_group(
            "Variant calling options",
            "Parameters and options for variant calling."
        )

        variant_grp.add_argument(
            "--variant-caller", type=str, choices=["pilon", "bcftools"],
            default="pilon",
            help="Variant caller to use for calling variants. Options are 'pilon' or "
            "'bcftools'. Note that the selected variant caller must be installed and "
            "available in the system PATH. Default is %(default)s."
        )

        variant_grp.add_argument(
            "--alignment-strictness", type=int, default=6,
            help="Strictness level for the alignment step within the variant calling. "
            "Higher values result in more stringent alignments. Default is %(default)s."
        )

        # ---------------- Variant filtering options ----------------

        filter_grp = parser.add_argument_group(
            "Variant filtering options",
            "Parameters and options for filtering variants based on the depth of "
            "coverage for the target strain."
        )

        filter_grp.add_argument(
            "--coverage-n-sites", type=int, default=100000,
            help="Number of sites to use for estimating the depth of coverage for the "
            "target strain. Default is %(default)s."
        )

        filter_grp.add_argument(
            "--min-depth", type=int, default=10,
            help="Minimum depth needed for a site to be considered in the filtering step. "
            "Sites with a depth of coverage lower than this value are flagged as LowCov in "
            "the output VCF. Default is %(default)s."
        )

        filter_grp.add_argument(
            "--min-alt-bc", type=int, default=10,
            help="Minimum number of reads supporting the alternate allele for an alternate "
            "allele to be called. Sites passing all other filters but with fewer "
            "than this number of reads supporting the alternate allele are assumed "
            "to have the reference allele. Default is %(default)s."
        )

        filter_grp.add_argument(
            "--min-ref-bc", type=int, default=10,
            help="Maximum number of reads supporting the reference allele for which "
            "CleanSweep automatically calls the alternate allele. A site must have "
            "both less than this number of reads supporting the reference allele and "
            "at least --min-alt-bc reads supporting the alternate allele to be called as "
            "a variant. Default is %(default)s."
        )

        filter_grp.add_argument(
            "--fit-method", type=str, choices=["fast", "mixture"], default="fast",
            help="Method to use for fitting the CleanSweep filters. Options are "
            "\"mixture\" or \"fast\". If \"mixture\", uses MCMC sampling to estimate "
            "the parameters of the depth of coverage distribution for the target "
            "strain. If \"fast\", uses maximum likelihood estimators based on the "
            "unique regions of the target strain. The \"mixture\" method is slightly "
            "more accurate but slower. Default is \"%(default)s\"."
        )

        filter_grp.add_argument(
            "--max-dispersion", type=float, default=70.0,
            help="Maximum allowed dispersion for the depth of coverage of the target "
            "strain. Ignored if `fit_method` is \"fast\". If `fit_method` is \"mixture\", "
            "this value is used to detect variants with low alternate allele base counts "
            "not reported by the variant caller. Further, it bounds the estimated "
            "dispersion. Default is %(default)s."
        )

        filter_grp.add_argument(
            "--fit_n_sites", type=int, default=200,
            help="Number of sites to use for fitting the CleanSweep filters. Ignored "
            "if --fit_method is \"fast\". Default is %(default)s."
        )

        filter_grp.add_argument(
            "--n-chains", type=int, default=4,
            help="Number of MCMC chains to run for fitting the CleanSweep filters. "
            "Ignored if --fit_method is \"fast\". Default is %(default)s."
        )

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
        variants: bool,
        method: Literal["fast", "mixture"],
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

        vcf_filter = VCFFilter(
            random_state = seed,
            method = method
        )
        
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
            overdispersion_bias = overdispersion_bias,
        )

        # Write the output VCF
        logging.debug(
            f'Writing filtered VCF to {str(outdir.joinpath("cleansweep.variants.vcf"))}...'
        )
        
        if variants:
            write_vcf(
                vcf = vcf_out,
                file = outdir.joinpath("cleansweep.variants.vcf"),
                header = VCF(str(input)).get_header(),
                chrom = prepare_dict['chrom'],
            )
        else:
            write_full_vcf(
                vcf = vcf_out,
                full_vcf = input,
                file = outdir.joinpath("cleansweep.variants.vcf"),
                header = VCF(str(input)).get_header(),
                chrom = prepare_dict['chrom'],
                min_dp = min_depth
            )
             
        # Save the filter and MCMC results
        vcf_filter.save(outdir.joinpath("cleansweep.filter.swp"))
        vcf_filter.save_samples(outdir.joinpath("cleansweep.posterior.swp"))
        
        logging.info("Done!")
        
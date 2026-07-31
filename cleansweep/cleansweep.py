"""
Contains a class to run CleanSweep end to end, given a set of reference strain genome 
FASTAs and plate sweep reads.

Author: Marco Teixeira
Email: mcarvalh@broadinstitute.org
"""

import logging
import joblib
import pandas as pd
from dataclasses import dataclass
from cleansweep.typing import File, Directory
from pathlib import Path
from typing import List, Union, Literal
from tempfile import TemporaryDirectory

from cleansweep.io import StrainGSTParser
from cleansweep.nucmer import NucmerAlignment
from cleansweep.call import VariantCaller
from cleansweep.filter import VCFFilter
from cleansweep.vcf import write_vcf, write_full_vcf, VCF

@dataclass
class CleanSweep:
    """
    Class to run CleanSweep end to end, given a set of reference strain genome FASTAs 
    and plate sweep reads.
    """

    def call(
        self,
        reads1: File,
        reads2: Union[File, None],
        reference: Union[File, str],
        background: List[File, None],
        output: Directory,
        prefix: str = "cleansweep",
        *,
        # Allows users to pass StrainGST results as input, instead of
        # providing a reference and background FASTAs.
        straingst_results: Union[List[File], None] = None,
        straingst_database: Union[Directory, None] = None,
        # Temporary directory and logging stuff
        tmp_dir: Union[None, Directory] = None,
        keep_tmp: bool = False,
        verbosity: int = 1,
        # Parameters for the `prepare` step
        min_mask_identity: float = 0.95,
        min_mask_length: int = 150,
        # Variant calling options
        variant_caller: Literal["bcftools", "pilon"] = "bcftools",
        alignment_strictness: int = 6,
        # SNV filtering options
        coverage_n_sites: int = 100000,
        min_depth: int = 10,
        min_alt_bc: int = 10,
        min_ref_bc: int = 10,
        fit_method: Literal["mixture", "fast"] = "mixture",
        # MCMC options for the mixture model
        max_dispersion: float = 70.0,
        fit_n_sites: int = 200,
        n_chains: int = 4,
        n_draws: int = 10000,
        n_burnin: int = 1000,
        mcmc_engine: Literal["pymc", "numpyro", "nutpie"] = "pymc",
        dispersion_bias: int = 1,
        variants: bool = False,
        # Compute resources and other options
        threads: int = 1,
        random_state: Union[int, None] = None
    ) -> Path:
        """
        Calls single-nucleotide variants for a target strain reference from plate
        sweep reads using CleanSweep.

        Parameters
        ----------
        
        reads1 : File
            Path to the first read file (FASTQ format).
        reads2 : File or None
            Path to the second read file (FASTQ format). If None, assumes single-end
            reads.
        reference : File or str
            Path to the reference genome FASTA file for the target strain. 
            Alternatively, if `straingst_results` is provided, this can be the name 
            of the target strain. CleanSweep will call variants against this 
            reference.
        background : List[File] or None
            List of paths to background genome FASTA files. These represent all other
            strains in the sample, except the target strain. If `straingst_results` 
            is provided, this can be None, and CleanSweep will infer the background
            strains from the StrainGST results.
        output : Directory
            Path to the output directory where the final filtered VCF will be written.
        prefix : str, optional
            Prefix for the output VCF file. The name of the final VCF will be 
            <prefix>.vcf. Default is "cleansweep".
        straingst_results : List[File] or None, optional
            List of paths to StrainGST result files. If provided, CleanSweep will use
            these results to infer the reference and background FASTAs, instead of
            requiring the user to provide them directly. Default is None.
        straingst_database : Directory or None, optional
            Path to the StrainGST database directory. Required if `straingst_results`
            is provided. Default is None.
        tmp_dir : Directory or None, optional
            Path to the temporary directory where intermediate files will be written.
            If None, a temporary directory will be created within the output directory.
            Default is None.
        keep_tmp : bool, optional
            If True, keeps the temporary files after the run. Default is False.
        verbosity : int, optional
            Verbosity level for logging. Ranges from 0 (debug) to 4 (error). Default 
            is 1 (info).
        min_mask_identity : float, optional
            Minimum mask identity for the alignment step within `cleansweep prepare`. 
            Regions of the background strain references aligning to the target 
            reference with at least this identity will be masked. Default is 0.95.
        min_mask_length : int, optional
            Minimum mask length for the alignment step within `cleansweep prepare`. 
            Regions of the background strain references aligning to the target 
            reference with at least this length will be masked. Default is 150.
        variant_caller : str, optional
            Variant calling method to use. Options are "bcftools" or "pilon".
            Default is "bcftools". The tool used must be installed and available 
            in the system PATH.
        alignment_strictness : int, optional
            Strictness level for the alignment step within the variant calling.
            Higher values result in more stringent alignments. Default is 6.
        coverage_n_sites : int, optional
            Number of sites to use for estimating coverage in the filtering step.
            Default is 100000.
        min_depth : int, optional
            Minimum depth for a site to be considered in the filtering step. Sites 
            with depth below this threshold will be filtered out. Default is 10.
        min_alt_bc : int, optional
            Minimum number of reads supporting the alternate allele for an alternate
            allele to be called. Sites passing all other filters but with fewer 
            than this number of reads supporting the alternate allele are assumed 
            to have the reference allele. Default is 10.
        min_ref_bc : int, optional
            Maximum number of reads supporting the reference allele for which CleanSweep
            automatically calls the alternate allele. A site must have both less than
            this number of reads supporting the reference allele and at least
            `min_alt_bc` reads supporting the alternate allele to be called as a
            variant. Default is 10.
        fit_method : str, optional
            Method to use for fitting the CleanSweep filters. Options are "mixture" or
            "fast". If "mixture", uses MCMC sampling to estimate the parameters of the 
            depth of coverage distribution for the target strain. If "fast", uses 
            maximum likelihood estimators based on the unique regions of the target 
            strain. The "mixture" method is slightly more accurate but slower. 
            Default is "mixture".
        max_dispersion : float, optional
            Maximum allowed dispersion for the depth of coverage of the target strain. 
            Ignored if `fit_method` is "fast". If `fit_method` is "mixture", this value 
            is used to detect variants with low alternate allele base counts not reported 
            by the variant caller. Further, it bounds the estimated dispersion. 
            Default is 70.0.
        fit_n_sites : int, optional
            Number of sites to use for fitting the CleanSweep filters. Ignored if
            `fit_method` is "fast". Default is 200.
        n_chains : int, optional
            Number of MCMC chains to use for fitting the CleanSweep filters. Ignored if
            `fit_method` is "fast". Default is 4.
        n_draws : int, optional
            Number of MCMC draws to use for fitting the CleanSweep filters. Ignored if
            `fit_method` is "fast". Default is 10000.
        n_burnin : int, optional
            Number of MCMC burn-in draws to use for fitting the CleanSweep filters.
            Ignored if `fit_method` is "fast". Default is 1000.
        mcmc_engine : str, optional
            MCMC engine to use for fitting the CleanSweep filters. Options are "pymc", 
            "numpyro", or "nutpie". Ignored if `fit_method` is "fast." Default is 
            "pymc".
        dispersion_bias : int, optional
            Controls the overdispersion prior for the target strain. More specifically,
            it is the value of the alpha and beta parameters of a Beta distribution, and
            greater values lead to a dispersion closer to 0.5. Ignored if `fit_method` 
            is "fast".  Default is 1.
        variants : bool, optional
            If True, writes only the filtered variants to the final VCF. If False,
            writes the entire filtered VCF. Default is False.
        threads : int, optional
            Number of threads to use for variant calling and filtering. Default is 1.
        random_state : int or None, optional
            Random seed for reproducibility. Default is None.

        Returns
        -------
        filtered_vcf : Path
            Path to the final filtered VCF file.
        """

        # Configure the logger
        self.__config_logger(verbosity)

        # Create the output directory if it doesn't exist
        self.outdir = self.__make_outdir(output)

        tmpdir = self.__make_tmpdir(tmp_dir)
        with TemporaryDirectory(
            dir = tmpdir, 
            prefix = "cleansweep_", 
            delete = (not keep_tmp)
        ) as tmp:

            # Get the reference and background FASTAs, either from the provided files
            # or from StrainGST results
            reference, background = self.__get_fastas(
                reference = reference,
                background = background,
                straingst_results = straingst_results,
                straingst_database = straingst_database
            )

            # ------------- CleanSweep prepare step -------------
            
            self.logger.info("Running CleanSweep prepare step...")
            self.prepare = NucmerAlignment(
                min_identity = min_mask_identity,
                min_length = min_mask_length
            )

            self.__prepare_dir = Path(tmp) / "prepare"
            self.__prepare_dir.mkdir(exist_ok=True)

            self.prepare.prepare(
                reference = reference,
                queries = background,
                outdir = self.__prepare_dir,
                keep_tmp = keep_tmp
            )

            # Path to the masked reference FASTA
            self.reference = self.__prepare_dir / "cleansweep.reference.fa"
            # Path to the cleansweep prepare SWP file
            self.prepare = self.__prepare_dir / "cleansweep.prepare.swp"

            # ------------ CleanSweep call step -------------
            
            self.logger.info("Running CleanSweep call step...")
            self.variant_caller = VariantCaller(threads=threads)

            # Path to the initial, unfiltered VCF file
            self.raw_vcf = Path(tmp) / "cleansweep.raw.vcf.gz"

            self.variant_caller.call(
                reads = self.__get_reads(reads1, reads2),
                reference = self.reference,
                output = self.raw_vcf,
                method = variant_caller,
                strictness = alignment_strictness
            )

            # ------------ CleanSweep filter step -------------

            self.logger.info("Running CleanSweep filter step...")
            self.vcf_filter = VCFFilter(
                method = fit_method,
                random_state = random_state
            )

            # Load the prepare file to get the gaps and SNPs for filtering
            self.logger.debug(f"Loading CleanSweep prepare file {str(self.prepare)}...")
            prepare_dict = joblib.load(self.prepare)

            # Create a temporary directory for the filter step
            self.__filter_dir = Path(tmp) / "filter"
            self.__filter_dir.mkdir(exist_ok=True)

            self.logger.debug(f"Calling VCFFilter.fit()...")

            filtered_vcf = self.vcf_filter.fit(
                vcf = self.raw_vcf,
                gaps = prepare_dict["gaps"],
                query = prepare_dict["chrom"][0],
                nucmer_snps = prepare_dict["snps"],
                tmp_dir = self.__filter_dir,
                n_coverage_sites = coverage_n_sites,
                min_depth = min_depth,
                min_alt_bc = min_alt_bc,
                min_ref_bc = min_ref_bc,
                max_overdispersion = max_dispersion,
                downsampling = fit_n_sites,
                chains = n_chains,
                draws = n_draws,
                burn_in = n_burnin,
                threads = threads,
                engine = mcmc_engine,
                overdispersion_bias = dispersion_bias
            )

            self.logger.debug(f"Filtering successful.")

            # Write the final filtered VCF to the output directory
            self.filtered_vcf = Path(output) / f"{prefix}.vcf"

            self.__write_final_vcf(
                vcf = filtered_vcf,
                header = VCF(str(self.raw_vcf)).get_header(),
                chrom = prepare_dict['chrom'],
                min_depth = min_depth,
                variants = variants,
                output = self.filtered_vcf,
                full_vcf = self.raw_vcf if not variants else None
            )

        self.logger.info(f"All clean! Final filtered VCF written to "
            f"{self.filtered_vcf}")
        
        return self.filtered_vcf

    def __config_logger(self, verbosity: int):
        """
        Configure the logger for the CleanSweep class.

        Parameters
        ----------
            verbosity: int
                Verbosity level for logging.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)

    def __make_outdir(self, output: Directory) -> Path:
        """
        Create the output directory if it doesn't exist.

        Parameters
        ----------
            output: Directory
                Path to the output directory.

        Returns
        -------
            outdir: Path
                Path to the output directory.
        """

        self.logger.debug(f"Creating output directory at {output}")
        outdir = Path(output)
        outdir.mkdir(parents=False, exist_ok=True)

        return outdir

    def __make_tmpdir(
        self, 
        tmp_dir: Union[None, Directory] = None
    ) -> Path:
        """
        Returns the path to the temporary directory. If tmp_dir is None, creates 
        a temporary directory within the output directory.

        Parameters
        ----------
            tmp_dir : None or Directory, optional
                Path to the temporary directory. If None, creates a temporary directory 
                within the output directory.
        
        Returns
        -------
            tmpdir: Path
                Path to the temporary directory.
        """

        if tmp_dir is None:
            self.logger.debug("No temporary directory provided. Creating a temporary "
                              "directory within the output directory.")
            tmp_dir = Path(self.outdir) / "tmp"
            
        else:
            self.logger.debug(f"Using provided temporary directory at {tmp_dir}")
            tmp_dir = Path(tmp_dir)
            if not tmp_dir.parent.exists():
                raise FileNotFoundError(f"Could not create a temporary directory at "
                    f"{str(tmp_dir)}. Parent directory does not exist.")

        return tmp_dir
    
    def __get_fastas(
        self,
        reference: Union[File, None],
        background: List[File, None],
        straingst_results: Union[List[File], None],
        straingst_database: Union[Directory, None]
    ) -> tuple[File, List[File]]:
        """
        Get the reference and background FASTAs, either from the provided files or 
        from StrainGST results.

        Parameters
        ----------
            reference : File or None
                Path to the reference genome FASTA file. If None, must provide 
                straingst_results.
            background : List[File] or None
                List of paths to background genome FASTA files. If None, must provide 
                straingst_results.
            straingst_results : List[File] or None
                List of paths to StrainGST result files. If provided, will be used to 
                get the reference and background FASTAs.
            straingst_database : Directory or None
                Path to the StrainGST database directory. Required if straingst_results 
                is provided.

        Returns
        -------
            reference : File
                Path to the reference genome FASTA file.
            background : List[File]
                List of paths to background genome FASTA files.
        """

        if not straingst_results is None:
            # Provided StrainGST results, so use them to

            if straingst_database is None:
                raise ValueError("Provided StrainGST results without setting a "
                    "StrainGST database.")
            if reference is not None or background is not None:
                self.logger.warning("Provided StrainGST results, but also provided a "
                    "reference and/or background FASTAs. The StrainGST results will be "
                    "used, and the provided reference/background FASTAs will be ignored.")

            straingst_parser = StrainGSTParser()

            # Find FASTAs for each detected strain
            fastas = straingst_parser.get_references(
                query = str(reference),
                straingst_strains = straingst_results,
                database_dir = straingst_database
            )

            ref_out, bckgd_out = fastas["query"], fastas["background"]

        else:
            # No StrainGST results provided, so use the provided reference and background
            if reference is None or background is None:
                raise ValueError("Must provide either a reference and background FASTAs "
                    "or StrainGST results.")

            ref_out, bckgd_out = reference, background

        return ref_out, bckgd_out

    def __get_reads(self, reads1: File, reads2: Union[File, None]) -> list[File]:
        """
        Get the list of read files, either single-end or paired-end.

        Parameters
        ----------
            reads1 : File
                Path to the first read file (FASTQ format).
            reads2 : File or None
                Path to the second read file (FASTQ format). If None, assumes 
                single-end reads.

        Returns
        -------
            reads : list[File]
                List of read files.
        """

        if reads2 is None:
            self.logger.debug("No second read file provided. Assuming single-end "
                              "reads.")
            return [reads1]
        else:
            self.logger.debug("Second read file provided. Assuming paired-end "
                              "reads.")
            return [reads1, reads2]

    def __write_final_vcf(
        self,
        vcf: pd.DataFrame,
        header: str,
        chrom: str,
        min_depth: int,
        variants: bool,
        output: File,
        full_vcf: Union[pd.DataFrame, None] = None,
    ):
        """
        Writes the final CleanSweep filtered VCF. 

        Parameters
        ----------
            vcf : pd.DataFrame
                DataFrame containing the filtered VCF data.
            header : str
                VCF header string.
            chrom : str
                Chromosome name.
            min_depth : int
                Minimum depth for filtering.
            variants : bool
                If True, writes only the filtered variants. If False, writes the full
                VCF, including information for the entire genome.
            output : File
                Path to the output VCF file.
            full_vcf : pd.DataFrame or None, optional
                DataFrame containing the full VCF data. Required if variants is False.
        """
        
        self.logger.debug(f"Writing final VCF to {output}...")

        if variants:

            self.logger.debug(f"Writing full VCF since 'variants' is True.")

            write_vcf(
                vcf = vcf,
                file = output,
                header = header,
                chrom = chrom,
            )
        else:

            if full_vcf is None:
                raise ValueError(
                    "Must provide full VCF DataFrame when writing full VCF.")

            self.logger.debug(f"Writing full VCF since 'variants' is False.")

            write_full_vcf(
                vcf = vcf,
                full_vcf = full_vcf,
                file = output,
                header = header,
                chrom = chrom,
                min_dp = min_depth
            )
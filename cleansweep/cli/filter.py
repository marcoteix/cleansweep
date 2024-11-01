import argparse
from typing import Union
from cleansweep.cli.commands import Subcommand
from cleansweep.vcf import VCF
from cleansweep.filter import VCFFilter

class FilterCmd(Subcommand):
    """Filters variants in a query strain, called with plate swipe data.

    Reads an output VCF file from Pilon and discards false positive variants
    caused by background reads. Outputs a VCF file with the passing sites.
    """

    def add_arguments(self, parser: argparse.ArgumentParser):
        
        parser.add_argument("input", type=str, help="Input Pilon VCF file.")
        parser.add_argument("output", type=str, help="Output VCF file.")
        
    def run(self, input: str, output: str, reference_ani: float, random_state: int, 
        expected_coverage: Union[int, None], coverage_fdr: float):

        # Read VCF
        vcf = VCF(input)
        
        # Filter
        vcf_filter = VCFFilter(
            reference_ani = reference_ani,
            random_state = random_state
        )
        filtered_vcf = vcf_filter.fit_filter(
            vcf.vcf,
            expected_coverage = expected_coverage,
            coverage_filter_params = {"p_threshold": coverage_fdr}
        )

        
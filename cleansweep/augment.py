import subprocess 
from dataclasses import dataclass
from cleansweep.typing import File
from cleansweep.vcf import VCF
from scipy.stats import nbinom
import pandas as pd
import logging

@dataclass
class AugmentVariantCalls:

    def augment(
        self,
        vcf: File,
        query: str,
        min_alt_bc: int,
        output: File
    ) -> pd.DataFrame:

        filter_expression = f"(INFO/BC[0] > {min_alt_bc} & REF != \"A\") | \
(INFO/BC[1] > {min_alt_bc} & REF != \"C\") | \
(INFO/BC[2] > {min_alt_bc} & REF != \"G\") | \
(INFO/BC[3] > {min_alt_bc} & REF != \"T\")"

        cmd = [
            "bcftools",
            "view",
            "-i",
            filter_expression,
            "-r",
            str(query),
            str(vcf)
        ]

        logging.info(
            f"Augmenting variant calls (alt base count > {min_alt_bc}) with the command \"{' '.join(cmd)}\"..."
        )

        with open(output, "w") as file:
            rc = subprocess.run(
                cmd,
                stdout = file 
            )

        logging.debug(
            f"Got return code {rc.returncode} (stdout: {rc.stdout})."
        )

        if rc.returncode:
            raise RuntimeError(
                f"Failed to augment the variant calls for file {str(vcf)} with the \
command \"{' '.join(cmd)}\". Got the following error: {rc.stdout}."
            )

        # Read augmented VCF
        try:
            vcf_df = VCF(output).read(
                chrom = query, 
                exclude = ["\'INFO/AC = 0 & REF!=\".\"\'"],
            )
            logging.debug(
                f"The augmented VCF has {len(vcf_df)} variants."
            )
        except pd.errors.EmptyDataError:
            raise RuntimeError(
                f"The augmented VCF file in {str(output)} is empty or has no SNVs."
            )
        
        return vcf_df


    def estimate_min_alt_bc(
        self,
        query_coverage: float,
        alpha: float = 0.01,
        overdispersion: float = .55
    ) -> int:
        
        return int(
            nbinom(
                query_coverage,
                overdispersion
            ).ppf(
                alpha
            )
        )
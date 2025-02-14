import subprocess 
from dataclasses import dataclass
from cleansweep.typing import File
from cleansweep.vcf import VCF
from scipy.stats import nbinom
import pandas as pd

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
            str(vcf),
            "-i",
            filter_expression,
            str(query),
            "-o",
            str(output),
            "-O",
            "z"
        ]

        rc = subprocess.run(cmd)

        if rc.returncode:
            raise RuntimeError(
                f"Failed to augment the variant calls for file {str(vcf)} with the \
command \"{' '.join(cmd)}\". Got the following error: {rc.stderr}."
            )
        
        # Read augmented VCF
        return VCF(output).read(
            chrom = query, 
            exclude = "\'INFO/AC = 0 & REF!=\".\"",
        )

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
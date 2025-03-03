#%%
from cleansweep.vcf import VCF
from cleansweep.typing import File, Directory
from dataclasses import dataclass
from typing import Union, List
from typing_extensions import Self
from pathlib import Path
import pandas as pd
import logging

FilePath = Union[str, Path]

@dataclass
class InputLoader:

    def load_vcf(self, vcf: File, query: str, **kwargs) -> pd.DataFrame:

        self.__path_exists(vcf)
        return VCF(vcf).read(
            chrom=query, 
            exclude = "\'INFO/AC = 0 & REF!=\".\" & ALT!=\".\"\'",
            **kwargs
        )

    def __path_exists(self, path: FilePath) -> None:

        if not Path(path).exists():
            raise FileNotFoundError(f"Could not find {str(path)}.")
    
    def load(
        self, 
        vcf: FilePath, 
        query: str, 
        *, 
        vcf_kwargs: dict = {}
    ) -> Self:

        self.vcf = self.load_vcf(vcf, query, **vcf_kwargs)

        return self
    
@dataclass
class StrainGSTParser:

    def get_references(
        self,
        query: str,
        straingst_strains: List[File],
        database_dir: Directory
    ) -> dict:
        
        # Read StrainGST output files and concatenate
        logging.debug(
            f"Reading the StrainGST output files in {', '.join([str(x) for x in straingst_strains])}..."
        )

        straingst = pd.concat(
            [
                pd.read_table(
                    x,
                    index_col = 0
                )
                for x in straingst_strains
            ]
        )

        logging.debug(
            f"Found {len(straingst)} strains in the StrainGST output files."
        )

        # Make sure the query is one of the detected strains
        if not query in straingst.strain.values:
            raise ValueError(
                f"The query strain ({query}) is not one of the strains detected by StrainGST."
            )

        if len(straingst) == 1:
            logging.warning(
                "StrainGST only detected one strain in the sample. CleanSweep may behave unexpectedly."
            )

        # Find the FASTA for each strain
        straingst = straingst.assign(
            fasta = straingst.strain \
                .apply(
                    lambda x: self.find_fasta(
                        strain = x,
                        database_dir = database_dir
                    )
                )
        )

        return {
            "query": straingst.loc[ 
                    straingst.strain.eq(query),
                    "fasta"
                ].iloc[0],
            "background": straingst.loc[ 
                    straingst.strain.ne(query),
                    "fasta"
                ].to_list()
        }

    def find_fasta(
        self,
        strain: str,
        database_dir: Directory
    ) -> str:

        database_dir = Path(database_dir)

        results = list(
            database_dir.glob(
                f"**/*{strain}.fa"
            )
        )

        if len(results) > 1:
            raise RuntimeError(
                f"Found multiple reference FASTA files for strain {strain}: {', '.join([str(x) for x in results])}."
            )
        elif not len(results):
            raise RuntimeError(
                f"Could not find a reference FASTA for strain {strain} in {str(database_dir)}. Gob pattern: \"**/*{strain}.fa\""
            )

        return str(results[0])
        
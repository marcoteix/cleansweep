import pandas as pd
from pathlib import Path

nucmer_header = ["pos", "ref", "alt", "pos_q", "dist_to_aln_end", "dist_to_end", 
    "n_repeats_r", "n_repeats_q", "strand_r", "strand_q", "ref_id", "query_id"]

class NucmerSNPs:

    def __init__(self):
        pass 

    def read(self, directory: str) -> pd.DataFrame:

        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"The directory {str(path)} does not exist.")
        if path.is_file():
            raise ValueError(f"The input path {path} provided is a file, not a directory.")
        
        # Find the SNP TSV files
        snp_files = path.glob("nucmer.*.snps.tsv")

        self.snps = pd.concat([pd.read_table(x, header=None, names=nucmer_header, skiprows=4) for x in snp_files])

        return self.snps
    
    def search(self, vcf: pd.DataFrame) -> pd.Series:

        if not hasattr(self, "snps"):
            raise RuntimeError("Tried calling \"search()\" without a prior call to \"read()\".")
        
        # Hash the reference VCF
        ref_hash = vcf.pos.astype(str)+"_"+vcf.ref+"_"+vcf.alt
        # Hash the query VCF
        query_hash = self.snps.pos.astype(str)+"_"+self.snps.ref+"_"+self.snps.alt

        return ref_hash.isin(query_hash)

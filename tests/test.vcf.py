from cleansweep.vcf import VCF, write_vcf
from cleansweep.__version__ import __version__

# Read test VCF
vcf = VCF(
    "test/data/test.vcf.gz"
)

vcf_df = vcf.read(
    chrom = "NZ_CP135691.1"
)

print(vcf_df.head())

# Write first 10 lines
write_vcf(
    vcf_df.iloc[:10],
    "test/outputs/vcf/test.vcf",
    chrom = "NZ_CP135691.1",
    ref = "test.fa",
    version = __version__
)
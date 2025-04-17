from cleansweep.vcf import VCF, write_vcf
from cleansweep.__version__ import __version__

# Read test VCF
vcf = VCF(
    "tests/data/pilon/test.small.vcf.gz"
)

vcf_df = vcf.read(
    chrom = "NZ_CP135691.1"
)

print(vcf_df.head())

# Write first 10 lines
write_vcf(
    vcf_df.iloc[:10],
    "tests/outputs/vcf/test.vcf",
    header = vcf.get_header(),
    chrom = "NZ_CP135691.1"
)
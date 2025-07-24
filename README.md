# CleanSweep

Finds strain-specific single nucleotide variants from plate swipe data.

## Usage

### Input data

Before calling variants with CleanSweep, we need to detect strains in the plate swipe. We recommend using StrainGE, although any other strain-detection tool should also work. To run CleanSweep, we will need:

- FASTQ files with plate swipe reads for alignment.
- A set of reference sequences (FASTA), one for each strain detected in the plate swipe.

CleanSweep can only call variants for one strain at a time, so one of the detected strains should be our target strain.

### Preparing a reference for alignment

Prepare a reference for alignment and a ``.swp`` file we will be using later with ``cleansweep prepare``:
```
cleansweep prepare \
    target_strain.fa \
    --background other_strain_1.fa other_strain_2.fa other_strain_3.fa \
    --output output/directory/ \
    --min-identity 0.95 \
    --min-length 150
```

We recommend setting the ``--min-length`` option to the insert size.

We should now have two files in the output directory: one named ``cleansweep.prepare.swp`` and a FASTA file named ``cleansweep.reference.fa``.

### Aligning plate swipe reads

Next, we can align the plate swipe reads to the FASTA created by CleanSweep prepare with BWA. This alignment should be strict, so we use high mismatch, clip, open, and extend penalties. For convenience, we offer a ``cleansweep align`` command that wraps BWA MEM.

```
cleansweep align \
    reads1.fq \
    reads2.fq \
    --reference cleansweep.reference.fa \
    --output cleansweep.alignment.bam
```

### Calling variants with Pilon

We can now call variants with Pilon, setting the ``--fix bases``, ``--nostrays`` and ``-duplicates`` options.

```
pilon \
    --genome cleansweep.reference.fa \
    --frags cleansweep.alignment.bam \
    --output samplename \
    --outdir pilon \
    --changes \
    --vcf \
    --fix bases \
    --nostrays \
    --duplicates

bcftools view \
    pilon/samplename.vcf \
    -o pilon/samplename.vcf.gz \
    -O z

bcftools index pilon/samplename.vcf.gz
```

### Filtering SNVs with CleanSweep

Many of the variants called by Pilon are ambiguous, as there likely are multiple strains in your plate swipe. We will use the CleanSweep ``filter`` command to figure out which of the SNVs detected by Pilon are trully present in the target strain.

```
cleansweep filter \
    pilon/samplename.vcf.gz \
    cleansweep.prepare.swp \
    cleansweep \
    --min-depth 10 \
    --min-alt-bc 10 \
    --min-ref-bc 20 \
    --downsample 200 \
    --max-overdispersion 0.7 \
    --overdispersion-bias 1 \
    --n-coverage-sites 100000 \
    --seed 23 \
    --n-chains 5 \
    --n-draws 100000 \
    --n-burnin 1000 \
    --threads 5 \
    --verbosity 4
```

This command will create four files in ``./cleansweep``:
- **cleansweep.variants.vcf** contains the filtering results as a VCF file. 
- **cleansweep.filter.swp** is a binary file containing information about the filtering process.
- **cleansweep.posterior.swp** is a binary file containing the results from the MCMC sampling within the ``cleansweep filter`` command.
- **cleansweep.filter.log** is a log file.

SNVs trully present in the target strain will have a value of ``PASS`` in the ``FILTER`` field in ``cleansweep.variants.vcf``.
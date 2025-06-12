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

    bwa mem \
    -t ~{cpu} \
    -O ~{open} \
    -E ~{extend} \
    -L ~{clip} \
    -U ~{unpaired} \
    -B ~{mismatch} \
    "${ref_genome}" \
    ~{read1} ~{read2}
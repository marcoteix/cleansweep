Usage
=====

Input data
----------

Before calling variants with CleanSweep, you need strain-detection results
(e.g. from StrainGE). You will need:

- FASTQ files with plate swipe reads for alignment.
- A set of reference sequences (FASTA), one per detected strain.

CleanSweep calls variants for one strain at a time, so one detected strain
should be designated as the target.

Preparing a reference
---------------------

.. code-block:: bash

   cleansweep prepare \
       target_strain.fa \
       --background other_strain_1.fa other_strain_2.fa \
       --output output/directory/ \
       --min-identity 0.95 \
       --min-length 150

This produces ``cleansweep.prepare.swp`` and ``cleansweep.reference.fa`` in
the output directory. Set ``--min-length`` to the insert size.

Aligning reads
--------------

.. code-block:: bash

   cleansweep align \
       reads1.fq \
       reads2.fq \
       --reference cleansweep.reference.fa \
       --output cleansweep.alignment.bam

Calling variants
----------------

Call variants with Pilon:

.. code-block:: bash

   pilon \
       --genome cleansweep.reference.fa \
       --frags cleansweep.alignment.bam \
       --output samplename \
       --outdir pilon \
       --changes --vcf --fix bases --nostrays --duplicates

   bcftools view pilon/samplename.vcf -o pilon/samplename.vcf.gz -O z
   bcftools index pilon/samplename.vcf.gz

Filtering SNVs
--------------

.. code-block:: bash

   cleansweep filter \
       pilon/samplename.vcf.gz \
       cleansweep.prepare.swp \
       cleansweep \
       --min-depth 10 \
       --threads 5

SNVs present in the target strain will have ``PASS`` in the ``FILTER`` field
of the output VCF.

Building a collection MSA
--------------------------

After running ``cleansweep filter`` on multiple plate swipes, build a multiple
sequence alignment with ``cleansweep collection``:

.. code-block:: bash

   cleansweep collection \
       sample1/cleansweep.variants.vcf.gz \
       sample2/cleansweep.variants.vcf.gz \
       sample3/cleansweep.variants.vcf.gz \
       --output collection.fasta \
       --reference cleansweep.reference.fa \
       --alpha 10 \
       --min-coverage 10

Use ``--exclude`` to remove divergent samples from the MSA instead of
cleaning their private SNPs:

.. code-block:: bash

   cleansweep collection ... --exclude --exclude-log excluded_samples.txt

Scaling to large collections
----------------------------

``--reference`` takes the reference FASTA used for variant calling — the
``cleansweep.reference.fa`` written by ``cleansweep prepare`` — optionally
gzip-compressed. Given it, CleanSweep reads only the variant and low-coverage
records from each VCF and patches them into a copy of the reference, rather than
walking every record of every VCF.

The alignment is then carried as a ``samples x variant sites`` array, so the
outlier scan, the consensus, and the private-SNP removal all scale with the
number of variant sites instead of with the genome length. This is what makes
collections of hundreds of samples practical; on a whole-site VCF it runs around
90 times faster and produces identical output.

Without ``--reference``, each VCF is converted to a full-length sequence and
positions with no record at all are left as ``N``. With it, the alignment spans
the whole reference and positions with no record take the reference base. For the
whole-site VCFs ``cleansweep filter`` writes by default the two modes agree; for
the sparse VCFs written by ``cleansweep filter --variants`` they do not, and the
reference-anchored result is the intended one.

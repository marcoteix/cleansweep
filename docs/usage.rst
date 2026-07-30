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
       --alpha 10 \
       --min-coverage 10

Use ``--exclude`` to remove divergent samples from the MSA instead of
cleaning their private SNPs:

.. code-block:: bash

   cleansweep collection ... --exclude --exclude-log excluded_samples.txt

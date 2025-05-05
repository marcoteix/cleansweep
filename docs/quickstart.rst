=================
Quickstart
=================

To detect strain-specific variants from plate swipe data with CleanSweep you will need the following:

- A set of reference genomes for strains in the plate swipe. We recommend using StrainGST_, but any strain detection tool will do.
- Define a strain you want to call variants in (i.e., the query strain). We will refer to the other strains in the plate swipe as background strains.
- Plate swipe read files.

###################################
Preparing a reference for alignment
###################################

Create a reference sequence with ``cleansweep prepare``. This removes shared regions from all but the query strain and finds a list of variant occuring between the references that are excluded later::

    cleansweep prepare \
        strain.query.fa \
        -b strain.1.background.fa strain.2.background.fa \
        -o output_directory

This prepares a reference for ``strain.query.fa``. The following files are written to ``output_directory``: 

- ``cleansweep.reference.fa``: the reference FASTA file.
- ``cleansweep.prepare.swp``: a file with information needed for ``cleansweep filter``.

###################################
Aligning reads and calling variants
###################################



.. _StrainGST: https://strainge.readthedocs.io/en/latest/

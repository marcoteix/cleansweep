#!/bin/bash

# Downsamples a VCF file by selecting random lines

bcftools view -H $1 | \
    head "-"$2 | \
    shuf -n $3
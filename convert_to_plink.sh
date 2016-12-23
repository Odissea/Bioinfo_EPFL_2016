#!/bin/bash
PATH_INPUT="./VCFfiles/"
PATH_OUTPUT="./OUTPUT/"

#Cleaning
rm -r $PATH_OUTPUT/*

for chrom in $PATH_INPUT/*.vcf.gz; do
    #Extract name of chromosome without path and extension
    chromName=$(basename "$chrom" | cut -f 1 -d '.')
    outputName="$PATH_OUTPUT$chromName"
    echo Processing chromosome $chromName ...
    ./plink --vcf $chrom --out $outputName
done

#Move the logs
mkdir $PATH_OUTPUT/LOGS
mv $PATH_OUTPUT/*.log $PATH_OUTPUT/LOGS

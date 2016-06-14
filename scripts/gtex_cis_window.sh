#!/bin/bash

# Usage:
# sh gtex_cis_window.sh gene_name chromosome transcription_start transcription_end

# Main Paths
MAIN=/projects/ps-jcvi/projects/mediator_was
PLINK=/home/kbhutani/tools/gwas/plink/plink
DATA=$MAIN/data
PDIR=$MAIN/processed/gtex

# File Paths 
VCF=$DATA/gtex/genotypes.new.header.vcf.gz

# Script generated variables
gene=$1
chr=$2
if [ $3 -gt 500000 ]
then
    begin=$(($3-500000))
else
    begin=0
fi
end=$(($4+500000))

# The output name for the plink-set containing cis-region around gene
odir=$PDIR/$chr/$gene
mkdir $odir
bfile=$odir/$gene
$plink --vcf $VCF --chr $chr --from-bp $begin --to-bp $end --make-bed \
       --keep-allele-order --out $bfile --snps-only --hwe midp 1e-5 \
       --geno 0.01 --maf 0.01 --allow-no-sex \
       --memory 8000 --prune


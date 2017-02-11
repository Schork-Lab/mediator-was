"""Fit LEAP and FastLMM on the WTCCC samples

This module provides an entry point to distribute the work per
phenotype/chromosome.

Author: Abhishek Sarkar <aksarkar@mit.edu>

"""
import argparse
import logging
import os
import sys

import fastlmm.association
import leap.leapMain
import leap.leapUtils
import numpy
import pandas
import pysnptools.snpreader.bed

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def fit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bfile', help='Plink dataset (imputed genotypes for GWAS)', required=True)
    parser.add_argument('--bfilesim', help='Plink dataset (array genotypes used for kinships)', required=True)
    parser.add_argument('--pve', help='Proportion of variance explained', type=float, required=True)
    parser.add_argument('-K', '--prevalence', help='Population case prevalence', type=float,
                        required=True)
    parser.add_argument('--chr', help='Exclude chromosome', type=int,
                        default=None)
    parser.add_argument('--prune', help='Threshold to prune related individuals', type=float, default=None)
    args = parser.parse_args()

    logger.info('Reading bfilesim')
    data = pysnptools.snpreader.bed.Bed(args.bfilesim, count_A1=False).read().standardize()
    keep = None
    if args.prune is not None:
        logger.info('Finding related individuals')
        keep = leap.leapUtils.findRelated(data, cutoff=args.prune)
    eigen_file = args.bfilesim
    if args.chr is not None:
        del data
        logger.info('Excluding chromosome {}'.format(args.chr))
        data = leap.leapUtils.getExcludedChromosome(args.bfilesim, args.chr)
        logger.info('Computing eigendecomposition excluding chromosome {}'.format(args.chr))
        eigen_file = '{}_{}'.format(args.bfilesim, args.chr)
    else:
        logger.info('Computing eigendecomposition')
    eigen = leap.leapMain.eigenDecompose(data, outFile=eigen_file)
    logger.info('Estimating liabilities')
    liabs = leap.leapMain.probit(data, '{}.pheno'.format(args.bfilesim), args.pve,
                                 args.prevalence, eigen, keepArr=keep)
    liabs_df = pandas.DataFrame(zip([x[1] for x in liabs['iid']], liabs['vals']))
    logger.info('Computing GWAS')
    if args.chr is not None:
        test = leap.leapUtils.getChromosome(args.bfile, args.chr)
    else:
        test = args.bfile
    result_df = leap.leapMain.leapGwas(bedSim=data, bedTest=test, pheno=liabs,
                                       h2=args.pve, eigenFile=eigen_file)
    logger.info('Writing results')
    result_df.to_csv('{}.{}.gwas'.format(args.bfile, args.chr),
                     sep="\t", header=False, index=False)
    liabs_df.to_csv('{}.{}.liab'.format(args.bfile, args.chr),
                    sep="\t", header=False, index=False)
    logger.info('Completed analysis')

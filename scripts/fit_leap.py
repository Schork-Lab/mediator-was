import pandas as pd
import numpy as np
import leap.leapUtils as leapUtils
import leap.leapMain as leapMain
from pysnptools.snpreader.bed import Bed
from fastlmm.util.stats import plotp
import fastlmm.association as association
import pylab

def fit(bfile, prevalence):
   phen_df = pd.read_table(bfile+'.fam', sep=" ", names=["study", "id", "m", "f", "g" ,"phen"])
   phenoFile = bfile+'.phen'
   phen_df['phen'] = phen_df['phen'].map(lambda x: x-1)
   phen_df[["study","id","phen"]].to_csv(phenoFile, sep="\t", header=False, index=False)
   chromosomes = xrange(1, 23)


   bed = Bed(bfile).read().standardize()
   # Step 1
   indsToKeep = leapUtils.findRelated(bed, cutoff=0.05)

   for chrom in chromosomes:
      bedExclude = leapUtils.getExcludedChromosome(bfile, chrom)
      eigenFile = 'temp_eigen.{}.npz'.format(bfile)
      eigen = leapMain.eigenDecompose(bedExclude, outFile=eigenFile)
      h2 = leapMain.calcH2(phenoFile, prevalence, eigen, keepArr=indsToKeep)
      liabs = leapMain.probit(bedExclude, phenoFile, h2, prevalence, eigen, keepArr=indsToKeep)
      liabs_df = pd.DataFrame(zip(map(lambda x: x[1], liabs['iid']1, liabs['vals']))
      liabs_df.to_csv(bfile+'.{}.{}.liab'.format(str(prevalence-int(prevalence))[1:], chrom), sep="\t", header=False, index=False)


if __name__ == "__main__":
   if len(sys.argv) < 3:
      print('Usage: python fit_leap.py bfile prevalence')
   else:
      fit(sys.argv[1], float(sys.argv[2]))
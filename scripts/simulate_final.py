import os
import glob
import pickle
import numpy.random as R
import pandas as pd
import seaborn as sns
import pymc3 as pm
import mediator_was.modeling.simulation_new as s
from mediator_was.processing.helpers import load_plink
import numpy.random
import mediator_was.modeling.simulation_new
import time

studies = ['/home/unix/kbhutani/compbio/twas/simulation/studies/top40_05_10_causal/top_40_1/study.pkl'
          '/home/unix/kbhutani/compbio/twas/simulation/studies/top40_05_10_causal/top_40_2/top_40_2_study.pkl', 
          '/home/unix/kbhutani/compbio/twas/simulation/studies/top40_05_10_causal/top_40_3/top_40_3_study.pkl',
          '/home/unix/kbhutani/compbio/twas/simulation/studies/top40_05_10_causal/top_40_4/top_40_4_study.pkl']
with pm.Model():
    studies = [pickle.load(open(study, 'rb')) for study in studies]


plink_dir = '/home/unix/kbhutani/compbio/gwas/genotypes'
plink = os.listdir(plink_dir)
plink = list(set([os.path.join(plink_dir, fn.split('.')[0])
                  for fn in plink if not fn.startswith('parse_genotypes')]))
loaded_plink = [load_plink(fn) for fn in plink]

numpy.random.seed(int(time.time()))

def random_gene():
    try:
      p_causal_eqtls = [0.01, 0.05, 0.1]
      seed = numpy.random.randint(low=0, high=10000)
      plink_idx = numpy.random.randint(0, len(plink)-1)
      p_causal = p_causal_eqtls[numpy.random.randint(0, 3)]
      haps = loaded_plink[plink_idx]
      plink_file = plink[plink_idx]
      gene = mediator_was.modeling.simulation_new.Gene("null", plink_file, p_causal_eqtls=p_causal, seed=seed, haps=haps)
    except:
      return None
    return gene

def associate(gene, study):
    seed = numpy.random.randint(low=0, high=10000)
    assoc = mediator_was.modeling.simulation_new.Association("null", gene, study, seed=seed)
    df = assoc.create_frequentist_df()
    return df

null_dfs = [pd.DataFrame() for _ in range(len(studies))]

for i in range(500):
  gene = random_gene()
  for i, study in enumerate(studies):
    null_association = associate(gene, study)
    null_dfs[i] = pd.concat([null_dfs[i]+null_association)

for i, null_df in enumerate(null_dfs):
  null_df.to_csv('{}_{}.tsv'.format(i, int(time.time()), sep="\t")
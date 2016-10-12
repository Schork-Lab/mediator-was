import os
import sys
import mediator_was.twas as T
import glob
import pandas as pd


def associate(gene_dir, study_prefix, out_prefix,
              gtex=True, rlog=True):
    study = T.Study(study_prefix)
    if gtex:
      gene = T.Gene(gene_dir)
      association = T.Association(gene, study)
      association.save(out_prefix+'.gtex')
      del association
      del gene
    if rlog:
      gene = T.Gene(gene_dir, gtex=False)
      association = T.Association(gene, study)
      association.save(out_prefix+'.rlog')      
    return

def aggregate(association_dir, prefix=None):
    def reader(fn):
       try: 
           return pd.read_table(fn, sep='\t')
       except:
            print('{} is empty.'.format(fn))
            return None

    for expr_type in ['rlog', 'gtex']:
      fns = glob.glob(os.path.join(association_dir,
                      '{}*{}.fstats.tsv'.format(prefix, expr_type)))
      print('{} frequentist files found.'.format(len(fns)))
      f_df = pd.concat([reader(fn)
                        for fn in fns if fn.find('aggregated') == -1])
      f_df.to_csv(".".join([prefix, '{}.aggregated.fstats.tsv'.format(expr_type)]),
                  sep='\t', index=False)

      fns = glob.glob(os.path.join(association_dir,
                      '{}*{}.bstats.tsv'.format(prefix, expr_type)))
      print('{} bayesian files found.'.format(len(fns)))
      b_df = pd.concat([b_reader(fn)
                      for fn in fns if fn.find('aggregated') == -1])
      b_df.to_csv(".".join([prefix, '{}.aggregated.bstats.tsv'.format(expr_type)]),
                  sep='\t', index=False)
    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python runner {associate, aggregate}")
        print("python runner.py associate gene_file study_file out_prefix")
        print("python runner.py aggregate association_dir out_prefix")
    else:
        if sys.argv[1] == "associate":
            print('Associating {} to {}'.format(sys.argv[2], sys.argv[3]))
            associate(sys.argv[2], sys.argv[3], sys.argv[4])
        elif sys.argv[1] == "aggregate":
            print('Running aggregate for {}'.format(sys.argv[2]))
            aggregate(sys.argv[2], sys.argv[3])
        else:
            print('Unrecognized command', sys.argv[1])

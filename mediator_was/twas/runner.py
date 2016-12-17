import os
import sys
import mediator_was.twas as T
import mediator_was.twas.bare as TBare
import glob
import pandas as pd


def calc_r2(gene_dir, gtex=True, rlog=True):
    if gtex:
      try:
        gene = T.Gene(gene_dir)
        gene._calc_r2()
        gene._save()
      except:
        pass
    if rlog:
      gene = T.Gene(gene_dir, gtex=False)
      gene._calc_r2()
      gene._save()
    return

def associate(gene_dir, study_prefix, out_prefix,
              gtex=True, rlog=True,):
    study = T.Study(study_prefix)
    if gtex:
      try:
        gene = T.Gene(gene_dir)
        gene._calc_r2()
        association = T.Association(gene, study)
        association.save(out_prefix+'.gtex')

        del association
        del gene
      except:
        pass
    if rlog:
      gene = T.Gene(gene_dir, gtex=False)
      association = T.Association(gene, study)
      association.save(out_prefix+'.rlog')      
    return

def associate_bare(gene_dir, study_prefix, out_prefix,
              gtex=True, rlog=True,):
    study = TBare.Study(study_prefix)
    if gtex:
      try:
        gene = TBare.Gene(gene_dir)
        association = TBare.Association(gene, study)
        association.save(out_prefix+'.gtex')
        del association
        del gene
      except:
        pass
    if rlog:
      gene = TBare.Gene(gene_dir, gtex=False)
      association = TBare.Association(gene, study)
      association.save(out_prefix+'.rlog')      
    return



def aggregate(association_dir, prefix=''):
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
      f_df.to_csv(os.path.join(association_dir,
                  '{}.{}.aggregated.fstats.tsv'.format(prefix, expr_type)),
                  sep='\t', index=False)

      fns = glob.glob(os.path.join(association_dir,
                      '{}*{}.bstats.tsv'.format(prefix, expr_type)))
      print('{} bayesian files found.'.format(len(fns)))
      b_df = pd.concat([reader(fn)
                      for fn in fns if fn.find('aggregated') == -1])
      b_df.to_csv(os.path.join(association_dir,
                  '{}.{}.aggregated.bstats.tsv'.format(prefix, expr_type)),
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
            associate_bare(sys.argv[2], sys.argv[3], sys.argv[4])
        elif sys.argv[1] == "aggregate":
            print('Running aggregate for {}'.format(sys.argv[2]))
            aggregate(sys.argv[2], sys.argv[3])
        else:
            print('Unrecognized command', sys.argv[1])

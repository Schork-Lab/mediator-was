import os
import sys
import mediator_was.twas as T
import glob
import pandas as pd


def associate(gene_dir, study_prefix, out_prefix):
    gene = T.Gene(gene_dir)
    study = T.Study(study_prefix)
    association = T.Association(gene, study)
    association.save(out_prefix)
    return


def aggregate(association_dir, prefix=None):
    fns = glob.glob(os.path.join(association_dir, '*.fstats.tsv'))
    print('{} frequentist files found.'.format(len(fns)))
    f_df = pd.concat([pd.read_table(fn, sep='\t')
                     for fn in fns if fn.find('aggregated') == -1])
    f_df.to_csv(".".join([prefix, 'aggregated.fstats.tsv']), sep='\t', index=False)
    fns = glob.glob(os.path.join(association_dir, '*.bstats.tsv'))

    def b_reader(fn):
       # Written because of initially not saving gene name
       df = pd.read_table(fn, sep='\t')
       if len(df.columns) == 8:
          df['gene'] = os.path.basename(fn).split('.')[0].split('_')[1]
          return df[[df.columns[0], 'gene']+list(df.columns[1:-1])]
       return df
    b_df = pd.concat([b_reader(fn)
                     for fn in fns if fn.find('aggregated') == -1])
    b_df.to_csv(".".join([prefix, 'aggregated.bstats.tsv']), sep='\t', index=False)
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

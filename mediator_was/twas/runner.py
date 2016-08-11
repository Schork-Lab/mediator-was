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
    f_df = pd.concat([pd.read_table(fn, sep='\t', index_col=[0, 1])
                     for fn in fns if fn.find('combined') != -1])
    f_df.to_csv('aggregated.fstats.tsv', sep='\t')
    fns = glob.glob(os.path.join(association_dir, '*.bstats.tsv'))
    b_df = pd.concat([pd.read_table(fn, sep='\t', index_col=[0, 1])
                     for fn in fns if fn.find('combined') != -1])
    b_df.to_csv(".".join([prefix, 'aggregated.bstats.tsv']), sep='\t')
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

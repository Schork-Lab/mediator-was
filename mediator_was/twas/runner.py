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

def associate_ts(gene_dir, study_prefix, out_prefix):
    gene = T.Gene(gene_dir)
    study = T.Study(study_prefix)
    association = T.Association(gene, study, associate=False)
    association.associate(**{'ts': True, 'joint': False})
    association.save(out_prefix)
    return


def permute(gene_dir, study_prefix,
            n_permutations, random_state,
            out_prefix):
    gene = T.Gene(gene_dir)
    study = T.Study(study_prefix)
    association = T.Association(gene, study, associate=False,
                                permute=(n_permutations, random_state))
    association.save(out_prefix)
    return


def aggregate(association_dir, prefix=None):
    fns = glob.glob(os.path.join(association_dir, '*.fstats.tsv'))
    print('{} frequentist files found.'.format(len(fns)))
    def f_reader(fn):
       try: 
           return pd.read_table(fn, sep='\t')
       except:
            print('{} is empty.'.format(fn))
            return None
    
    f_df = pd.concat([f_reader(fn)
                     for fn in fns if fn.find('aggregated') == -1])
    f_df.to_csv(".".join([prefix, 'aggregated.fstats.tsv']),
                sep='\t', index=False)
    fns = glob.glob(os.path.join(association_dir, '*.bstats.tsv'))

    def b_reader(fn):
        '''
          Written because of initially not saving gene name
        '''
        try:
            df = pd.read_table(fn, sep='\t')
        except:
            print('{} is empty.'.format(fn))
            return None
        if len(df.columns) == 8:
            df['gene'] = os.path.basename(fn).split('.')[0].split('_')[1]
            return df[[df.columns[0], 'gene'] + list(df.columns[1:-1])]
        return df

    b_df = pd.concat([b_reader(fn)
                      for fn in fns if fn.find('aggregated') == -1])
    b_df.to_csv(".".join([prefix, 'aggregated.bstats.tsv']),
                sep='\t', index=False)
    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python runner {associate, aggregate}")
        print("python runner.py associate gene_file study_file out_prefix")
        print("python runner.py aggregate association_dir out_prefix")
        print("python runner.py permutate gene_file study_file n_permutations random_state out_prefix")
    else:
        if sys.argv[1] == "associate":
            print('Associating {} to {}'.format(sys.argv[2], sys.argv[3]))
            associate(sys.argv[2], sys.argv[3], sys.argv[4])
        elif sys.argv[1] == "associate_ts":
            print('Two stage associating {} to {}'.format(sys.argv[2], sys.argv[3]))
            associate_ts(sys.argv[2], sys.argv[3], sys.argv[4])
        elif sys.argv[1] == "aggregate":
            print('Running aggregate for {}'.format(sys.argv[2]))
            aggregate(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == "permute":
            print('Permuting for {} to {}'.format(sys.argv[2], sys.argv[3]))
            permute(sys.argv[2], sys.argv[3],
                      int(sys.argv[4]), int(sys.argv[5]),
                      sys.argv[6])
        else:
            print('Unrecognized command', sys.argv[1])

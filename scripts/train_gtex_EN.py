import sys, os
import pandas as pd
import numpy
from sklearn.linear_model import ElasticNetCV
from sklearn.utils import resample
from plinkio import plinkfile


def reverse_alleles(x):
    if x == 0:
        return 2
    elif x == 2:
        return 0
    else:
        return x
reverse_alleles = numpy.vectorize(reverse_alleles)


def load_alleles_df(plink_file):
    plink_file = plinkfile.open(plink_file)
    sample_ids = map(lambda x: x.iid, plink_file.get_samples())
    allele_ids = map(lambda x: x.name, plink_file.get_loci())
    alleles = numpy.array(list(plink_file)).T
    alleles = reverse_alleles(alleles) 
    allele_df = pd.DataFrame(alleles, columns=allele_ids, index=sample_ids)
    return allele_df


def main(gene, chromosome, out_dir, n_bootstraps=50):
    '''
    Fit ElasticNet bootstraps using Yongjin's rlog transformed files.
    '''
    covariates_df = pd.read_table('/projects/ps-jcvi/projects/mediator_was/data/Whole_Blood_Analysis.covariates.txt',
                                 index_col=0).T
    data_df = pd.read_table('/projects/ps-jcvi/projects/mediator_was/data/tis53/chr{}.data.gz'.format(chromosome),
                            header=None, )
    gene_df = pd.read_table('/projects/ps-jcvi/projects/mediator_was/data/tis53/chr{}.info.gz'.format(chromosome), 
                           names=['chromosome', 'start', 'end', 'ensemblid', 'name'])
    sample_df = pd.read_table('/projects/ps-jcvi/projects/mediator_was/data/tis53/tis53.samples.txt.gz', header=None)

    data_df.columns = gene_df.name
    data_df.index = sample_df[0].map(lambda x: "-".join(x.split('-')[:2]))
    pcs_df = pd.read_table('/projects/ps-jcvi/projects/mediator_was/data/tis53/chr{}.pcs.tsv'.format(chromosome),
                          index_col=0)

    plink_file = "/projects/ps-jcvi/projects/mediator_was/results/{}/{}".format(gene, gene)
    allele_df = load_alleles_df(plink_file)

    ## Create dataframes
    endo_df = pd.concat([allele_df, covariates_df[['C1', 'C2', 'C3', 'gender']], pcs_df], axis=1).dropna()
    exo_df = data_df[[gene]].dropna()
    samples = list(set(endo_df.index).intersection(exo_df.index))

    ## Output for use for other programs
    samples_file = plink_file+'.samples'
    with open(samples_file, 'w') as OUT:
        for sample in samples:
            OUT.write(sample+"\n")

    covariates_file = plink_file+'.covariates.tsv'
    pd.concat([covariates_df[['C1', 'C2', 'C3', 'gender']], pcs_df], axis=1).ix[samples].to_csv(covariates_file, sep="\t", 
                                                                                              index=False, header=False)

    alleles_file = plink_file+'.alleles.tsv'
    allele_df.ix[samples].to_csv(alleles_file, sep="\t", index=False, header=False)

    allele_id_file = plink_file+'.locinames'
    allele_name_df = pd.DataFrame(allele_df.columns, columns=['id'])
    allele_name_df['chromosome'] = allele_name_df['id'].map(lambda x: x.split('_')[0])
    allele_name_df['position'] = allele_name_df['id'].map(lambda x: x.split('_')[1])
    allele_name_df['ref'] = allele_name_df['id'].map(lambda x: x.split('_')[2])
    allele_name_df['alt'] = allele_name_df['id'].map(lambda x: x.split('_')[3])
    allele_name_df['gene'] = gene
    allele_name_df.to_csv(allele_id_file, sep="\t", index=False)

    phen_file = plink_file+'.phen'
    exo_df.ix[samples].to_csv(phen_file, sep="\t", header=False, index=False)



    coef_dfs = []
    for i in range(25):
        b_samples = resample(samples)
        model = ElasticNetCV(max_iter=10000)
        model.fit(endo_df.ix[b_samples], exo_df.ix[b_samples].values.ravel())
        coef_df = pd.DataFrame(model.coef_[:allele_df.shape[1]], index=allele_df.columns, columns=['beta'])
        coef_df['chromosome'] = coef_df.index.map(lambda x: x.split('_')[0])
        coef_df['position'] = coef_df.index.map(lambda x: x.split('_')[1])
        coef_df['ref'] = coef_df.index.map(lambda x: x.split('_')[2])
        coef_df['alt'] = coef_df.index.map(lambda x: x.split('_')[3])
        coef_df['gene'] = gene
        coef_df['bootstrap'] = i
        coef_dfs.append(coef_df)


    coef_df = pd.concat(coef_dfs)
    coef_df.to_csv(os.path.join(out_dir, gene+'.tsv'), sep="\t")
    return coef_df

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python train_gtex_EN.py gene chromosome out_dir')
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

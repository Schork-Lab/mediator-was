import os
import sys
import pandas as pd
import sklearn.linear_model
from sklearn.utils import resample


imputed_loci = pd.read_table("/projects/ps-jcvi/projects/mediator_was/data/RA.bim", header=None)
imputed_loci = set(imputed_loci[0].astype(str)+'_'+imputed_loci[3].astype(str))

def refit_gene(gene, imputed_loci=imputed_loci):
    def create_coeff_df(model, allele_df):
        coef_df = pd.DataFrame(model.coef_[:allele_df.shape[1]], index=allele_df.columns, columns=['beta'])
        coef_df['chromosome'] = coef_df.index.map(lambda x: x.split('_')[0])
        coef_df['position'] = coef_df.index.map(lambda x: x.split('_')[1])
        coef_df['ref'] = coef_df.index.map(lambda x: x.split('_')[2])
        coef_df['alt'] = coef_df.index.map(lambda x: x.split('_')[3])
        coef_df['gene'] = gene
        coef_df = coef_df[coef_df.beta != 0]
        return coef_df

    print('Fitting {}'.format(gene))
    coef_dfs = []
    main_dir = "/projects/ps-jcvi/projects/mediator_was/training/results/{}/".format(gene)

    samples = pd.read_table(os.path.join(main_dir, gene+'.samples'), header=None)[0]

    covariates_df = pd.read_table(os.path.join(main_dir, gene+".covariates.tsv"),
                             names=['C1', 'C2', 'C3', 'gender']+['PC{}'.format(i) for i in range(15)])
    covariates_df.index = samples

    loci = pd.read_table(os.path.join(main_dir, gene+'.locinames'))
    loci['tag'] = loci['chromosome'].astype(str)+'_'+loci['position'].astype(str)
    loci['overlapping'] = loci['tag'].map(lambda x: True if x in imputed_loci else False)

    allele_df = pd.read_table(os.path.join(main_dir, gene+".alleles.tsv"), header=None)
    allele_df = allele_df.applymap(lambda x: 0 if x == 3 else x)
    allele_df.index = samples
    allele_df.columns = loci['id']
    allele_df = allele_df[loci[loci['overlapping']]['id']]
    print(allele_df.shape) 

    expression = pd.read_table(os.path.join(main_dir, gene+".phen"), header=None)
    expression.index = samples

    design = pd.concat([allele_df, covariates_df], axis=1)
    l1_ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    full_model = sklearn.linear_model.ElasticNetCV(l1_ratio=l1_ratio_range, max_iter=10000)
    full_model.fit(design, expression.values.ravel())
    
    coef_df = create_coeff_df(full_model, allele_df)
    coef_df['bootstrap'] = 'full'
    coef_dfs.append(coef_df)

    with open(os.path.join(main_dir, "bootstrap_params_v2.txt"), "w") as OUT:
        OUT.write("L1 ratio: {} \n".format(full_model.l1_ratio_))
        OUT.write("Alpha: {} \n".format(full_model.alpha_))

    
    for i in range(5):
        b_samples = resample(samples, replace=False, n_samples=300)
        model = sklearn.linear_model.ElasticNet(alpha=full_model.alpha_, 
                                                 l1_ratio=full_model.l1_ratio_, 
                                                 max_iter=10000)
        #model = sklearn.linear_model.ElasticNetCV(n_alphas=100, n_jobs=36, 
        #                                          l1_ratio=l1_ratio_range, 
        #                                          max_iter=10000)
        model.fit(design.ix[b_samples], expression.ix[b_samples].values.ravel())
        coef_df = create_coeff_df(model, allele_df)
        coef_df['bootstrap'] = i
        coef_dfs.append(coef_df)

    return pd.concat(coef_dfs)

def main(gene_list_file, out_file):
    with open(gene_list_file) as IN:
        genes = IN.read().split()

    with open(gene_list_file+'.notfound', 'w') as OUT:
        coef_dfs = []
        for gene in genes:
            #try:
                gene_df = refit_gene(gene)
                coef_dfs.append(gene_df)
            #except:
                OUT.write(gene+'\n')
            #    continue
    coef_df = pd.concat(coef_dfs)
    coef_df.to_csv(out_file, sep="\t")
    return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retrain_gtex_EN.py gene_list.txt out_file")
    else:
        main(sys.argv[1], sys.argv[2])

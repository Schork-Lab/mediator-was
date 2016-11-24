import os
import sys
import pickle
import mediator_was.twas.simulation as s
import pymc3 as pm
import numpy.random
import time
import pandas as pd


def simulate_null_associations(in_file,
                               n=20,
                               plink_dir='/home/unix/kbhutani/compbio/gwas/genotypes'):
    

    # Random numpy seed
    numpy.random.seed(int(time.time()))
    
    # Load Plink parameters
    plink = os.listdir(plink_dir)
    plink = list(set([os.path.join(plink_dir, fn.split('.')[0])
                      for fn in plink if not fn.startswith('parse_genotypes')]))
    p_causal_eqtls = [0, 0.01, 0.05, 0.1]
    
    # Load all the studies
    with open(in_file) as IN:
        studies, out_prefix = zip(*[line.rstrip().split() for line in IN]) 
    with pm.Model():
        studies = [pickle.load(open(study, 'rb')) for study in studies]
    
    # Initialize dataframes of statistics
    f_stats = [pd.DataFrame() for _ in range(len(studies))]
    b_stats = [pd.DataFrame() for _ in range(len(studies))]
    
    # Simulate n genes and associate with each study
    for i in range(n):
        print('Simulating gene {}'.format(i))
        seed = numpy.random.randint(low=0, high=10000)
        plink_idx = numpy.random.randint(0, len(plink)-1)
        p_causal = p_causal_eqtls[numpy.random.randint(0, 4)]
        plink_file = plink[plink_idx]
        try:
            gene = simulate_gene("null", plink_file, p_causal_eqtls=p_causal, seed=seed)
        except:
            continue
        for j, study in enumerate(studies):
            print('Associating')
            association = s.association("null", gene, study)
            f_stats[j] = pd.concat([f_stats[j], association.create_frequentist_df()])
            b_stats[j] = pd.concat([b_stats[j], association.create_bayesian_df()])
            del association
        del gene

    # Save statistics
    for i, pref in enumerate(out_prefix):
        fn = '{}_{}.fstats.tsv'.format(pref, time.time())
        f_stats[i].to_csv(fn, sep='\t')
        b_stats[i].to_csv(fn.replace('fstats', 'bstats'), sep='\t')

    return f_stats, b_stats


def simulate_gene(gene_name, plink_file, out_file=None, *args, **kwargs):
    gene = s.Gene(gene_name, plink_file, *args, **kwargs)
    if out_file is not None:
        print('Writing out file: {}'.format(out_file))
        with open(out_file, 'wb') as f:
            pickle.dump(gene, f)
    return gene


def simulate_study(study_name, gene_list_file, out_file=None, *args, **kwargs):
    genes = []
    with open(gene_list_file) as IN:
        for line in IN:
            with pm.Model():
                gene = pickle.load(open(line.rstrip(), 'rb'))
            genes.append(gene)
    study = s.Study(study_name, genes, *args, **kwargs)
    with open(out_file, 'wb') as f:
        pickle.dump(study, f)
    return


def associate(association_name, gene_file, study_file, out_file=None):
    with pm.Model():
        gene = pickle.load(open(gene_file, 'rb'))
    study = pickle.load(open(study_file, 'rb'))
    association = s.Association(association_name, gene, study)
    if out_file is not None:
        association.save(out_file.replace('.pkl', ''))
    # with open(out_file, 'wb') as f:
    #     pickle.dump(association, f)
    return association


def power(association_dir, out_file):
    power = s.Power(association_dir=association_dir)
    with open(out_file, 'wb') as f:
        pickle.dump(power, f)
    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: ")
        print("python simulation_runner.py simulate_gene gene_name plink_file out_file {optional p_causal_eqtls}")
        print("python simulation_runner.py simulate_study study_name gene_list_file out_file {optional seed}")
        print("python simulation_runner.py associate association_name gene_file study_file out_file")
        print("python simulation_runner.py simulate_null in_file")
        print("infile: study.pkl\tout_prefix")
        print("python simulation_runner.py power association_dir out_file")
    else:
        if sys.argv[1] == "simulate_gene":
            print('Simulating gene {}'.format(sys.argv[2]))
            p_causal_eqtls = float(sys.argv[5]) if len(sys.argv) > 5 else .1
            print('p_causal_eqtls {}'.format(p_causal_eqtls))
            simulate_gene(sys.argv[2], sys.argv[3], sys.argv[4],
                          p_causal_eqtls=p_causal_eqtls)
        elif sys.argv[1] == "simulate_study":
            print('Simulating study phenotype '.format(sys.argv[2]))
            seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
            n_samples = int(sys.argv[6]) if len(sys.argv) > 6 else 5000
            print('Seed: {}'.format(seed))
            print('Number of samples: {}'.format(n_samples))
            simulate_study(sys.argv[2], sys.argv[3], sys.argv[4],
                           seed=seed, n_samples=n_samples)
        elif sys.argv[1] == "associate":
            print('Associating {} to {}'.format(sys.argv[3], sys.argv[4]))
            associate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        elif sys.argv[1] == "power":
            print('Running power for {}'.format(sys.argv[2]))
            power(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == "simulate_null":
            print("Running null associations for {}".format(sys.argv[2]))
            simulate_null_associations(sys.argv[2])
        else:
            print('Unrecognized command', sys.argv[1])
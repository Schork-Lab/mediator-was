
import sys
import pickle
import mediator_was.twas.simulation as s
import pymc3 as pm


def simulate_gene(gene_name, plink_file, out_file=None, *args, **kwargs):
    gene = s.Gene(gene_name, plink_file, *args, **kwargs)
    print('Writing out file: {}'.format(out_file))
    with open(out_file, 'wb') as f:
        pickle.dump(gene, f)
    return


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


def associate(association_name, gene_file, study_file, out_file):
    with pm.Model():
        gene = pickle.load(open(gene_file, 'rb'))
    study = pickle.load(open(study_file, 'rb'))
    association = s.Association(association_name, gene, study)
    association.save(out_file.replace('.pkl', ''))
    # with open(out_file, 'wb') as f:
    #     pickle.dump(association, f)
    return


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
        else:
            print('Unrecognized command', sys.argv[1])
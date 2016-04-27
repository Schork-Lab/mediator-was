
import sys
import pickle
import mediator_was.modeling.simulation_new as s
import pymc3 as pm

def simulate_gene(gene_name, plink_file, out_file=None, *args, **kwargs):
    gene = s.Gene(gene_name, plink_file)
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
    study = s.Study(study_name, genes)
    with open(out_file, 'wb') as f:
        pickle.dump(study, f)
    return


def associate(association_name, gene_file, study_file, out_file):
    print(gene_file)
    with pm.Model():
        gene = pickle.load(open(gene_file, 'rb'))
    study = pickle.load(open(study_file, 'rb'))
    association = s.Association(association_name, gene, study)
    association.save_frequentist(out_file.replace('.pkl','')+'.f_associations')
    with open(out_file, 'wb') as f:
        pickle.dump(association, f)
    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python runner.py {gene, phenotype, association}")
        print("python.py runner.py simulate_gene gene_name plink_file out_file")
        print("python.py runner.py simulate_study study_name gene_list_file out_file")
        print("python.py runner.py associate association_name gene_file study_file out_file")
    else:
        if sys.argv[1] == "simulate_gene":
            print('Simulating gene {}'.format(sys.argv[2]))
            simulate_gene(sys.argv[2], sys.argv[3], sys.argv[4])
        elif sys.argv[1] == "simulate_study":
            print('Simulating study phenotype '.format(sys.argv[2]))
            seed = sys.argv[5] if len(sys.argv) > 5 else 0
            simulate_study(sys.argv[2], sys.argv[3], sys.argv[4], seed)
        elif sys.argv[1] == "associate":
            print('Associating {} to {}'.format(sys.argv[3], sys.argv[4]))
            associate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        else:
            print('Unrecognized command', sys.argv[1])
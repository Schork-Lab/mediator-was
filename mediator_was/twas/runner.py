
import sys
import pickle
import mediator_was.twas as T

def associate(association_name, gene_dir, study_prefix, out_file):
    gene = T.Gene(gene_dir)
    study = T.Study(study_prefix)
    association = T.Association(gene, study)
    with open(out_file, 'wb') as f:
        pickle.dump(association, f)
    with open(out_file.replace('.pkl', '.stats.pkl'), 'wb') as f:
        pickle.dump([association.b_stats, association.f_stats], f)
    return


def power(association_dir, out_file):
    power = s.Power(association_dir=association_dir)
    with open(out_file, 'wb') as f:
        pickle.dump(power, f)
    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python runner {gene, phenotype, association}")
        print("python runner.py simulate_gene gene_name plink_file out_file {optional p_causal_eqtls}")
        print("python runner.py simulate_study study_name gene_list_file out_file {optional seed}")
        print("python runner.py associate association_name gene_file study_file out_file")
        print("python runner.py power association_dir out_file")
    else:
        if sys.argv[1] == "associate":
            print('Associating {} to {}'.format(sys.argv[3], sys.argv[4]))
            associate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        elif sys.argv[1] == "power":
            print('Running power for {}'.format(sys.argv[2]))
            power(sys.argv[2], sys.argv[3])
        else:
            print('Unrecognized command', sys.argv[1])
from mediator_was.twas.bare import Gene, Study, Association
import argparse
import logging
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', help='Path to gene directory', required=True)
    parser.add_argument('--study', help='Path to study vcf prefix', required=True)
    parser.add_argument('--out', help='Path to out files', required=True)
    parser.add_argument('--rlog', help='Use RLog transformed values, default', 
                        dest='gtex', action='store_false')
    parser.add_argument('--gtex', help='Use GTEx-Norm values', 
                        dest='gtex', action='store_true')
    parser.add_argument('--min_inclusion', help='Minimum inclusion probability for BAY-TS, default: 0.5',
                        type=float, default=0.5)
    parser.add_argument('--max_missing', help='Maximum missing genotypes for exclusion for study genotypes, default: 0.1',
                        type=float, default=0.1)
    parser.set_defaults(gtex=False)
    args = parser.parse_args()

    logger.info('Loading gene {}'.format(args.gene))
    gene = Gene(args.gene, args.gtex)

    logger.info('Loading Study {}'.format(args.study))
    study = Study(args.study)

    logger.info('Associating')
    association = Association(gene, study, min_p_inclusion=args.min_inclusion, missing_filter=args.max_missing)
    association.save(args.out)


if __name__ == "__main__":
    main()
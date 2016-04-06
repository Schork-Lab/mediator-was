import gzip
import sys
from collections import defaultdict
import pandas as pd
import numpy as np


def _convert_chrom(chrom):
    chrom = str(chrom)
    if chrom.startswith('chrom'):
        chrom = chrom[5:]
    elif chrom.startswith('chr'):
        chrom = chrom[3:]
    if chrom == 'X':
        chrom = '23'
    elif chrom == 'Y':
        chrom = '24'
    try:
        chrom = int(chrom)
    except:
        chrom = 25
    # assert chrom <= 24, 'Unrecognized chromosome'
    return chrom


def _opener(file_name):
    if file_name.endswith('.gz'):
        return gzip.open(file_name, 'rt')
    else:
        return open(file_name)

def _max_locus(locus1, locus2):
    '''
    locus: (chromosome, position)
    '''

    chrom1, chrom2 = _convert_chrom(locus1[0]), _convert_chrom(locus2[0])
    pos1, pos2 = int(locus1[1]), int(locus2[1])
    if chrom1 > chrom2:
        return locus1
    elif (chrom1 == chrom2) and (pos1 > pos2):
        return locus1
    elif (chrom1 == chrom2) and (pos1 == pos2):
        return 'Equal'
    else:
        return locus2


def _is_match(loci1, loci2):
    ref1, alt1 = loci1
    ref2, alt2 = loci2
    return (ref1 == ref2) and (alt1 == alt2)


def _get_samples_vcf(fn):
    with _opener(fn) as IN:
        for line in IN:
            # CHROM-POS-ID-REF-ALT-QUAL    FILTER  INFO    FORMAT SAMPLES
            if line.startswith('#'):
                samples = line.rstrip()
            else:
                break
    samples = samples.split()[9:]
    return samples


def _parser_vcf(line, alt_allele='1'):
    def _calculate_alleles(genotype, allele=None):
        '''
        Helper function to calculate number of alleles
        '''
        alleles = defaultdict(int)
        alleles[genotype[0]] += 1
        alleles[genotype[2]] += 1
        if allele:
            return alleles[allele]
        return alleles

    if line.startswith('#'):
        return '0', '0', '0', '0', [0]

    data = line.rstrip().split()
    chromosome = data[0]
    position = data[1]
    ref = data[3]
    alt = data[4]
    genotypes = [sample.split(':')[0] for sample in data[9:]]
    allele_counts = map(_calculate_alleles,
                        genotypes)
    alt_counts = [count[alt_allele]
                  for count in allele_counts]
    return chromosome, position, ref, alt, alt_counts


def _parser_oxstats(line):
    data = line.rstrip().split()
    chromosome = data[0]
    position = data[2]
    ref = data[3]
    alt = data[4]
    alt_counts = []
    for i in np.arange(5, len(data)-2, 3):
        alt_count = 1*float(data[i+1])
        alt_count += 2*float(data[i+2])
        alt_counts.append(alt_count)
    return chromosome, position, ref, alt, alt_counts

def _parser_dosage(line):
    if line.startswith('CHR'):
        return '0', '0', '0', '0', [0]
    data = line.rstrip().split()
    chromosome = data[0]
    position = data[3]
    ref = data[4]
    alt = data[5]
    ignore_missing = lambda x: int(x) if x != 'NA' else 0
    alt_counts = map(ignore_missing, data[6:])
    return chromosome, position, ref, alt, alt_counts

def _get_samples_dosage(genotype_file):
    with _opener(genotype_file) as IN:
        line = IN.next()
        samples = line.rstrip().split()[6:]
    return samples

def _get_samples_oxstats(genotype_file):
    sample_file = genotype_file.replace('.gen', '.samples')
    sample_df = pd.read_table(sample_file, sep=" ")
    samples = sample_df[sample_df.columns[1]]
    return samples

def _reverse(ref, alt):
    reverse_map = {'C': 'G', 'A': 'T', 'T': 'A', 'G':'C'}
    return reverse_map[ref], reverse_map[alt]

def stream(weight_file, genotype_file, genotype_filetype):
    '''
    Predicts expression of genes found in weights_file.

    weights_file requires the following columns:
        gene, beta, chromosome, position, rsid, ref, alt

    Assumes that the vcf/genotype_file is coordinate sorted.
    '''

    # Prepare weights
    weights = pd.read_table(weight_file, sep="\t")
    print(weights.columns)
    required_columns = set(['gene', 'beta',
                            'chromosome', 'position'])
    column_match = len(required_columns.intersection(weights.columns))
    assert column_match == 4, "All required columns not found"
    weights['chromosome'] = weights.chromosome.map(_convert_chrom)
    weights = weights.sort(['chromosome', 'position'])
    weights.index = range(len(weights))



    # Prepare genotype file
    if genotype_filetype == 'vcf':
        parser = _parser_vcf
        samples = _get_samples_vcf(genotype_file)
    elif genotype_filetype == 'oxstats':
        parser = _parser_oxstats
        samples = _get_samples_oxstats(genotype_file)
    elif genotype_filetype == 'dosage':
        parser = _parser_dosage
        samples = _get_samples_dosage(genotype_file)
    predicted_expression = defaultdict(lambda: np.zeros(len(samples)))

    # Sync to entry 0 for weights db
    db_index = 0
    entry = weights.ix[db_index]
    entry_locus = (entry['chromosome'], entry['position'])

    # Iterate through genotype file
    with _opener(genotype_file) as IN:
        for line in IN:
            chrom, position, gen_ref, gen_alt, alt_alleles = parser(line)
            gen_locus = (chrom, position)
            max_locus = _max_locus(gen_locus, entry_locus)

            if max_locus == entry_locus:  # Sync genotype file
                continue

            while max_locus == gen_locus:  # Sync weights db
                db_index += 1
                if db_index >= len(weights):
                    break
                entry = weights.ix[db_index]
                entry_locus = (entry['chromosome'], entry['position'])
                max_locus = _max_locus(gen_locus, entry_locus)

            while max_locus == 'Equal':  # Synced
               
                weights_ref, weights_alt = entry['ref'], entry['alt']
                if not _is_match((gen_ref, gen_alt),
                                 (weights_ref, weights_alt)):
                    # print('Non-matching reference and alt:', file=sys.stderr)
                    # print(chrom, position, entry['rsid'], file=sys.stderr)
                    # print('Genotype ref: {}, alt {}'.format(gen_ref, gen_alt), file=sys.stderr)
                    # print('Weights ref: {}, alt {}'.format(weights_ref, weights_alt), file=sys.stderr)
                    # See if order is switched 
                    if _is_match((gen_ref, gen_alt), _reverse(weights_ref, weights_alt)):
                        # print("Reverse strand at {}:{}".format(chrom, position), file=sys.stderr)
                        pass
                    elif _is_match((gen_alt, gen_ref), (weights_ref, weights_alt)):
                        # Switch the order
                        #print('Switching reference and alt at {}:{}, {}:{} and {}:{}'.format(chrom, position, gen_ref, gen_alt, weights_ref, weights_alt), file=sys.stderr)
                        # Switch genotypes
                        if genotype_filetype == 'vcf':
                            data = parser(line, alt_allele='0')
                            alt_alleles = data[-1]
                        else:
                            # TODO: Figure out how to deal with missing for this case.
                            alt_alleles = 2 - alt_alleles  # Assumes no missing
                    else:
                        # print('Skipping entry', file=sys.stderr)
                        # print(chrom, position, entry['rsid'], file=sys.stderr)
                        # print('Genotype ref: {}, alt {}'.format(gen_ref, gen_alt), file=sys.stderr)
                        # print('Weights ref: {}, alt {}'.format(weights_ref, weights_alt), file=sys.stderr)
                        break  # Break out of while loop


                gene = entry['gene']
                beta = entry['beta']
                contribution = np.array(alt_alleles)*beta
                predicted_expression[gene] += contribution

                # Update
                db_index += 1
                if db_index >= len(weights):
                    break
                entry = weights.ix[db_index]
                entry_locus = (entry['chromosome'], entry['position'])
                max_locus = _max_locus(gen_locus, entry_locus)

                if db_index % 15000 == 0:
                    print(db_index, 'processed out of', len(weights))

        predicted_weights = pd.DataFrame.from_dict(predicted_expression,
                                                   orient='index')
        predicted_weights.columns = samples
        return predicted_weights

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: predict.py weights_file genotype_file genotype_filetype{vcf, oxstats, dosage} out_file')
        exit()
    df = stream(sys.argv[1], sys.argv[2], sys.argv[3])
    df.to_csv(sys.argv[4], sep="\t")
    

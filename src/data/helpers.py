import gzip
from collections import defaultdict

import numpy as np
import pandas as pd

def calculate_alleles(genotype, allele=None):
    '''
    Helper function to calculate number of alleles
    '''
    alleles = defaultdict(int)
    alleles[genotype[0]] += 1
    alleles[genotype[2]] += 1
    if allele:
        return alleles[allele]
    return alleles


def gz_aware_open(fn):
    if fn.endswith('.gz'):
        return gzip.open(fn)
    else:
        return open(fn)


def convert_chrom(chrom):
    chrom = str(chrom)
    if chrom.startswith('chrom'):
        chrom= chrom[5:]
    elif chrom.startswith('chr'):
        chrom = chrom[3:]
    if chrom == 'X':
        chrom = '23'
    elif chrom == 'Y':
        chrom = '24'
    chrom = int(chrom)
    assert chrom <= 24, 'Unrecognized chromosome'
    return chrom

def get_max_locus(locus1, locus2):
    '''
    locus: (chromosome, position)
    '''

    chrom1, chrom2 = convert_chrom(locus1[0]), convert_chrom(locus2[0])
    pos1, pos2 = int(locus1[1]), int(locus2[1])

    if chrom1 > chrom2:
        return locus1
    elif (chrom1 == chrom2) and (pos1 > pos2):
        return locus1
    elif (chrom1 == chrom2) and (pos1 == pos2):
        return 'Equal'
    else:
        return locus2

is_match = lambda ref1, ref2, alt1, alt2: (ref1 == ref2) and (alt1 == alt2)


def stream_predict(df, vcf_file):
    '''
    Predicts expression of genes found in df.

    df requires the following columns:
        gene, beta, chromosome, position
    '''
    required_columns = set(['gene', 'beta', 'chromosome', 'position'])
    column_match = len(required_columns.intersection(df.columns))
    assert column_match == 4, "All required columns not found"

    df['chromosome'] = df.chromosome.map(convert_chrom)
    df = df.sort(['chromosome', 'position'])
    df.index = range(len(df))

    with gz_aware_open(vcf_file) as IN:
        for line in IN:
            # CHROM-POS-ID-REF-ALT-QUAL    FILTER  INFO    FORMAT SAMPLES
            if line.startswith('#'):
                samples = line.rstrip()
            else:
                break
    samples = samples.split()[9:]

    predicted_expression = defaultdict(lambda: np.zeros(len(samples)))
    entries_not_found = []
    incorrect_entries = []
    switched_entries = []

    db_index = 0
    entry = df.ix[db_index]
    entry_locus = (entry['chromosome'], entry['position'])
    with gz_aware_open(vcf_file) as IN:
        for line in IN:

            if db_index >= len(df):
                break

            if line.startswith('#'):
                continue


            data = line.rstrip('\n').split()
            chrom, position = data[0], data[1]
            vcf_locus = (chrom, position)
            max_locus = get_max_locus(vcf_locus, entry_locus)
            if max_locus == entry_locus:  # Continue to next position
                continue
            elif max_locus == vcf_locus:  # Entry locus not found, sync db
                while get_max_locus(vcf_locus, entry_locus) == vcf_locus:
                    entries_not_found.append(entry)
                    db_index += 1
                    if db_index >= len(df):
                        break
                    entry = df.ix[db_index]
                    entry_locus = (entry['chromosome'], entry['position'])
            # Equal
            while get_max_locus(vcf_locus, entry_locus) == 'Equal':
                vcf_ref, vcf_alt = data[3], data[4]
                db_ref, db_alt = entry['refAllele'], entry['alt']
                alt_allele = '1'
                if not is_match(vcf_ref, db_ref, vcf_alt, db_alt):
                    #print 'Non-matching reference and alt:'
                    #print chrom, position, entry['rsid']
                    #print vcf_ref, vcf_alt
                    #print db_ref, db_alt
                    # See if order is switched
                    if not is_match(vcf_ref, db_alt, vcf_alt, db_ref):
                        incorrect_entries.append(entry)
                        break # Break out of while loop
                    else:
                        # Switch the order
                        #print 'Switching reference and alt:'
                        #print chrom, position
                        switched_entries.append(entry)
                        alt_allele = '0'
                genotypes = [sample.split(':')[0] for sample in data[9:]]
                allele_counts = map(calculate_alleles,
                                    genotypes)
                alt_counts = [count[alt_allele]
                              for count in allele_counts]
                gene = entry['gene']
                beta = entry['beta']
                contribution = np.array(alt_counts)*beta
                predicted_expression[gene] += contribution

                # Update
                db_index += 1
                if db_index >= len(df):
                    break
                entry = df.ix[db_index]
                entry_locus = (entry['chromosome'], entry['position'])


        print 'Total', db_index
        print 'Not Found', len(entries_not_found)
        print 'Incorrect', len(incorrect_entries)
        print 'Switched', len(switched_entries)
        predicted_df = pd.DataFrame.from_dict(predicted_expression,
                                              orient='index')
        predicted_df.columns = samples
        return predicted_df, entries_not_found, incorrect_entries, switched_entries

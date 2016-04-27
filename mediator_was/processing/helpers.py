'''
Helpers for processing files.
'''
import pandas as pd
from plinkio import plinkfile

def load_plink(plink_prefix):
    """
    Load alleles from a plink file.
    """
    def reorder_alleles(allele):
        '''
        Reorder alleles based on alternative count, since plinkio assigns allele counts based on reference. 
        Assign missing alleles as 0 for the purposes of simulation.
        '''
        if allele == 0:
            allele = 2
        if allele == 2:
            allele = 0
        if allele == 3: # Missing, but should be okay for simulation.
            allele = 0
        return allele
    genotypes = plinkfile.open(plink_prefix)
    haps = pd.DataFrame.from_records(genotypes).ix[1:]
    haps.index = genotypes.get_loci()
    haps.columns = genotypes.get_samples()
    haps = haps.applymap(reorder_alleles)
    #haps = haps.T
    return haps

def load_hapgen(hap_file):
    """Load in a HapGen2 generated haplotypes file.
    """
    haps = pd.read_table(hap_file, sep=" ", header=None)
    return haps
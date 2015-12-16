'''
Script to aggregate BSLMM studies by Gusev et. al.

Note that the config.yaml should be updated to include BSLMM information.
Please copy over from config.template.yaml the new information.
'''
import os
import yaml
import mediator_was.processing.bslmm as bslmm


full_path = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(full_path,
                                     '../config.yaml')))
if __name__ == "__main__":
    studies = bslmm.parse_studies()
    for study, study_df in studies.iteritems():
        fn = 'BSLMM_%s_aggregated.tsv' % study
        study_df.to_csv(os.path.join(config['analysis']['dir'],
                                     fn),
                        sep='\t')

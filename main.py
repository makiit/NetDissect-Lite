import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model_file', metavar='N', type=str,
                    help='Model weights ')
args = parser.parse_args()

fo = FeatureOperator()
model = loadmodel(hook_feature,args["model_file"])

############ STEP 1: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=model)

for layer_id,layer in enumerate(settings.FEATURE_NAMES):
############ STEP 2: calculating threshold ############
    thresholds = fo.quantile_threshold(features[layer_id],savepath="quantile.npy")

############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id],thresholds,savepath="tally.csv")

############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    if settings.CLEAN:
        clean()
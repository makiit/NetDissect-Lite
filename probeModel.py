import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
import argparse
import numpy as np
import os



iters = [10,20,30,40,50]
for it in iters:
    print("Iteration ",it)
    path = "resnet18_cifar100_featuremaps/"
    max_path = path+"cifar100_actual_Max_features-iter_%d.mmap"%it
    whole_path = path+"cifar100_actual_Whole_features-iter_%d.mmap"%it
    fo = FeatureOperator(path)
#     print("Loading model file ",args.model_file)
#     model = loadmodel(hook_feature,args.model_file)
    size_features =  np.load("test_feature_size.npy")[0]
    ############ STEP 1: feature extraction ###############
    features= np.memmap(whole_path, dtype=float, mode='r', shape=tuple(size_features))
    maxfeature = np.memmap(max_path, dtype=float, mode='r', shape=tuple(size_features[:2]))
    print(np.shape(features))
    print(np.shape(maxfeature))
    ############ STEP 2: calculating threshold ############
    thresholds = fo.quantile_threshold(features,savepath="quantile_iter%d.npy"%it)

    ############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features,thresholds,savepath="tally_iter-%d.csv"%it)
    html_path = path+"iter%d"%it
    os.mkdir(html_path)
    ############ STEP 4: generating results ###############
    generate_html_summary(fo.data,html_path, "layer4",
                              tally_result=tally_result,
                              maxfeature=maxfeature,
                              features=features,
                              thresholds=thresholds)
    print("OUT")
#     if settings.CLEAN:
#         clean(path)
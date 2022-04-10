import torch
import torchvision
import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean

import os
from torch.autograd import Variable as V
from PIL import Image
# from scipy.misc import imresize
import numpy as np
import torch
import settings
import time
import util.upsample as upsample
import util.vecquantile as vecquantile
import multiprocessing.pool as pool
from loader.data_loader import load_csv
from loader.data_loader import SegmentationData, SegmentationPrefetcher

features_blobs = []
start_unit = 180
end_unit = 190
def hook_feature(module, input, output):
#     print("Hooking module named ",module.name)
    features_blobs.append(output.data.cpu().numpy()[:,start_unit:end_unit,:,:])


iters  = [10,20,30,40,50]
for it in iters:
    model_file = "checkpoints/resnet18-%d-regular.pth"%it
    # model_file = "zoo/resnet18_places365.pth.tar"
    # path = "iter200"
    # fo = FeatureOperator(path)
    print("Loading model file ",model_file)
    # model = loadmodel(hook_feature,model_file)
    # model = loadmodel(hook_feature,model_file)
    checkpoint = torch.load(model_file)
    model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = torch.nn.Identity()
    state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    data = SegmentationData(settings.DATA_DIRECTORY, categories=settings.CATAGORIES)
    loader = SegmentationPrefetcher(data,categories=['image'],once=True,batch_size=settings.BATCH_SIZE)
    mean = [109.5388,118.6897,124.6901]
    num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size
    print(num_batches)
    print(model._modules.get("layer4"))
    model._modules.get("layer4").register_forward_hook(hook_feature)
    print(len(loader.indexes))
    maxfeatures = [None] * len(settings.FEATURE_NAMES) #Max activation in the channel (broden_size,no.of channels)
    wholefeatures = [None] * len(settings.FEATURE_NAMES) #All the activations (broden_size,no.of channels,h,w)
    features_size = [None] * len(settings.FEATURE_NAMES)
    memmap=True
    for batch_idx,batch in enumerate(loader.tensor_batches(bgr_mean=mean)):
            del features_blobs[:]
            torch.cuda.empty_cache()
            input = batch[0]
            batch_size = len(input)
            print('extracting feature from batch %d / %d' % (batch_idx+1, num_batches))
            input = torch.from_numpy(input[:, ::-1, :, :].copy())
            input.div_(255.0 * 0.224)
            if settings.GPU:
                input = input.cuda()
            input_var = V(input,volatile=True)
            logit = model.forward(input_var)  
    #         if(batch_idx>2):break
            #Initializing the feature variables
            print("Feature Blobs length ",len(features_blobs))
            print("First blob",np.shape(features_blobs[0]))
            if maxfeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    print("Initializing %d"%i)
                    size_features = (len(loader.indexes), feat_batch.shape[1])
                    if memmap:
                        maxfeatures[i] = np.memmap('cifar100_actual_Max_features-iter_%d.mmap'%it,dtype=float,mode='w+',shape=size_features)
                    else:
                        maxfeatures[i] = np.zeros(size_features)

            #Initializing the all feature variable
            if len(feat_batch.shape) == 4 and wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (
                    len(loader.indexes), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap('cifar100_actual_Whole_features-iter_%d.mmap'%it, dtype=float, mode='w+', shape=size_features)
                    else:
                        wholefeatures[i] = np.zeros(size_features)

            np.save("test_feature_size.npy", features_size)
            start_idx = batch_idx*settings.BATCH_SIZE
            end_idx = min((batch_idx+1)*settings.BATCH_SIZE, len(loader.indexes))
            for i, feat_batch in enumerate(features_blobs):
                if len(feat_batch.shape) == 4:
                    wholefeatures[i][start_idx:end_idx] = feat_batch
                    maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)
                elif len(feat_batch.shape) == 3:
                    maxfeatures[i][start_idx:end_idx] = np.max(feat_batch, 2)
                elif len(feat_batch.shape) == 2:
                    maxfeatures[i][start_idx:end_idx] = feat_batch



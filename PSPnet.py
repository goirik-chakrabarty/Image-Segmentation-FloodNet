# %%
# tensorflow==2.2.0
# keras==2.3.1

import os

import tensorflow as tf
from tensorflow.keras import layers as layer
import segmentation_models as sm
# %%
# Model creation using segmentation_model
# %%
# Define model
def get_model(h,w,optimiser):
    '''
    This is a PSPNet model with the following specifications:-
    backbone = resnet101
    classes = 10
    input_dim = output_dim = hxw = 713x713
    activation = softmax
    loss = Not fixed
    metric = mIoU
    encoder_weights = imagenet
    '''

    model = sm.PSPNet(backbone_name='resnet101',
        input_shape=(h,w,3),
        classes=10,
        activation='softmax'
    )
    model.compile(
        optimiser,
        loss=sm.losses.categorical_focal_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    model.summary()

    return model

# %%
def preprocessing(x):
    return sm.get_preprocessing(x)

# %%
# get_model(384,384,'adam')
# # %%
# from keras.models import load_model
# # %%
# model_name='PSPNet_12img_25epoch_384'

# # %%
# PSPNet = load_model('saved_models/'+model_name+'.hdf5', compile=True)
# # %%

# %%
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt 
from random import randint
import cv2

import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam

import PSPnet
# %%
def general_preprocessing(train_dir, n_classes, height, width, img_height, img_width, h_n, w_n, seed, num_imgs, resize_dim):    
    train_flooded_imgs=[]
    train_flooded_img_dir = train_dir+"/Labeled/Flooded/image"
    imgs = os.listdir(train_flooded_img_dir)
    imgs.sort()
    imgs = imgs[:num_imgs]             # To be removed, truncating the number of images loaded for tesing purposes

    # for loading all training images in a numpy array
    for img_path in imgs:
        img = cv2.imread("/".join((train_flooded_img_dir,img_path)),1)
        img = img[0:img_height, 0:img_width, :]
        img = cv2.resize(img, resize_dim)

        i = 0
        j = 0
        y = 0

        for i in range(h_n):
            x=0
            for j in range(w_n):
                img_crop = img[y:(y+height), x:(x+width), :]
                train_flooded_imgs.append(img_crop)
                x+=width
            y+=height

    train_flooded_imgs = np.array(train_flooded_imgs)

    train_flooded_masks = []
    train_flooded_mask_dir = train_dir+"/Labeled/Flooded/mask"
    masks = os.listdir(train_flooded_mask_dir)
    masks.sort()
    masks = masks[:num_imgs]           # To be removed, truncating the number of masks loaded for tesing purposes

    # for loading all training masks in a numpy array
    for mask_path in masks:
        mask = cv2.imread("/".join((train_flooded_mask_dir,mask_path)),0)
        mask = mask[0:img_height, 0:img_width]
        mask = cv2.resize(mask, resize_dim)

        i = 0
        j = 0
        y = 0

        for i in range(h_n):
            x=0
            for j in range(w_n):
                mask_crop = mask[y:(y+height), x:(x+width)]
                train_flooded_masks.append(mask_crop)
                x+=width
            y+=height

    train_flooded_masks = np.array(train_flooded_masks)

    # Sanity Check
    rn = randint(0,(num_imgs*h_n*w_n)-1)
    img_san = train_flooded_imgs[rn]
    mask_san = train_flooded_masks[rn]
    plt.imshow(img_san)
    print(rn, img_san.shape)

    # Re-encoding the masks because the segmentation models library accepts only labels from [0, ... , n]
    labelencoder = LabelEncoder()
    n, h, w = train_flooded_masks.shape
    train_masks_reshaped = train_flooded_masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_flooded_masks = train_masks_reshaped_encoded.reshape(n, h, w)

    # Cleaning up memory
    del train_masks_reshaped
    del train_masks_reshaped_encoded

    # Expanding dimensions because libraries expect and RGB channel.
    train_flooded_masks = np.expand_dims(train_flooded_masks, axis=3)

    # categorical loss function or other deep learning classification loss functions expect the input to be one hot encoded
    n_classes = len(np.unique(train_flooded_masks))     # this line can be removed when sure about the number of classes
    flood_masks_cat = to_categorical(train_flooded_masks, num_classes=n_classes, dtype='uint8')
    train_flooded_masks_cat = flood_masks_cat.reshape((train_flooded_masks.shape[0],train_flooded_masks.shape[1],train_flooded_masks.shape[2],n_classes))

    # Test-Train split for training
    X_train, X_test, y_train, y_test = train_test_split(train_flooded_imgs, train_flooded_masks_cat, test_size=0.2, random_state=seed)
    X_train = X_train/255
    X_test = X_test/255

    return X_train, X_test, y_train, y_test

def model_preprocessing(X_train, X_test):
    preprocessor = PSPnet.preprocessing('resnet101')

    X_train = preprocessor(X_train)
    X_test = preprocessor(X_test)

    return X_train, X_test

def poly_lr(epoch, learning_rate=0.0001, power=0.9):
    lrate = learning_rate*((1 - epoch/epochs)**power)
    return lrate
# %%
# Global Variables

# Model specifications
num_imgs = 40
epochs = 25
power = 0.9
learning_rate = 0.0001
momentum = 0.9
dropout = 0.2
train_grid_size = 768       # 384 or 768
resize_dim = (2*train_grid_size, 2*train_grid_size)
dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = dt +'PSPNet'

# General Prepocessing specifications
seed = 5
np.random.seed = seed
wd = os.getcwd()                # working directory
train_dir = wd+"/data/Train"    # training directory
n_classes=10                    # number of classes
height = train_grid_size
width = train_grid_size
img_height = 3000
img_width = 4000

h_n = int(resize_dim[0]/height)
w_n = int(resize_dim[0]/width)
# %%
X_train, X_test, y_train, y_test = general_preprocessing(train_dir, n_classes, height, width, img_height, img_width, h_n, w_n, seed, num_imgs,resize_dim)

# %%
X_train, X_test = model_preprocessing(X_train, X_test)
# %%
# optimiser = SGD(learning_rate=learning_rate, momentum=momentum)
optimiser = Adam(learning_rate=learning_rate)
model = PSPnet.get_model(height,width,optimiser,dropout)
# %%
# Training the model
tens_dir = wd+"/logs/tensorboard/"
csv_logfile = wd+"/logs/csvlogs/"+model_name+".csv"
model_path = wd+"/saved_models/"+model_name+"epoch{epoch:02d}-{val_iou_score:.2f}.hdf5"

lr_rate = keras.callbacks.LearningRateScheduler(poly_lr)


callbacks = [
    keras.callbacks.TensorBoard(log_dir=tens_dir+model_name, histogram_freq=1),
    # keras.callbacks.EarlyStopping(patience=3,monitor="val_iou_score", verbose=1),
    keras.callbacks.CSVLogger(csv_logfile),
    keras.callbacks.ModelCheckpoint(model_path, monitor="val_iou_score", verbose=1, save_best_only=True, mode='max'),
    # lr_rate,
    ]
# %%
filename = open('model_logs.csv', 'a')
properties = [
    dt,
    model_name,
    seed,
    train_grid_size,
    num_imgs,
    epochs,
    str(optimiser),
    learning_rate,
    momentum,
    power,
    dropout,
    ]
new_line = ','.join(str(p) for p in properties)
filename.write("\n"+new_line+",")
filename.close()

history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
            )

# %%
#######################################################
############## TESTING MODEL BEGINS HERE ##############
#######################################################
# Note: X_test, y_test needed in memory. So run the inittial blocks this loading of model. 

from keras.models import load_model
from keras.metrics import MeanIoU

import tensorflow as tf
from tensorflow.keras import layers as layer
import segmentation_models as sm
# %%
PSPNet = load_model(
    'saved_models/'+model_name+'.hdf5',
    custom_objects = {
        'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss,
        'iou_score': sm.metrics.IOUScore(),
    }
    )
# %%
y_pred=PSPNet.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
# %%
n_classes = 10
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())
# %%

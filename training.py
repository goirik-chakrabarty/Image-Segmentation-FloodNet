# %%
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt 
import cv2

import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


import PSPnet
# %%
def general_preprocessing(train_dir, n_classes, height, width, img_height, img_width, h_n, w_n, seed, num_imgs):    
    train_flooded_imgs=[]
    train_flooded_img_dir = train_dir+"/Labeled/Flooded/image"
    imgs = os.listdir(train_flooded_img_dir)
    imgs.sort()
    imgs = imgs[:num_imgs]             # To be removed, truncating the number of images loaded for tesing purposes

    # for loading all training images in a numpy array
    for img_path in imgs:
        img = cv2.imread("/".join((train_flooded_img_dir,img_path)),1)
        img = img[0:img_height, 0:img_width, :]

        i = 0
        j = 0
        y = 0

        for i in range(h_n):
            x=0
            for j in range(w_n):
                train_flooded_imgs.append(img[y:(y+height), x:(x+width), :])
                x+=width
            y+=height

        # train_flooded_imgs.append(img)

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

        i = 0
        j = 0
        y = 0

        for i in range(h_n):
            x=0
            for j in range(w_n):
                train_flooded_masks.append(mask[y:(y+height), x:(x+width)])
                x+=width
            y+=height

        # train_flooded_masks.append(mask)

    train_flooded_masks = np.array(train_flooded_masks)

    # Re-encoding the masks because the segmentation models library accepts only labels from [0, ... , n]
    labelencoder = LabelEncoder()
    n, h, w = train_flooded_masks.shape

    # print(np.unique(train_flooded_masks))

    train_masks_reshaped = train_flooded_masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_flooded_masks = train_masks_reshaped_encoded.reshape(n, h, w)

    # print(np.unique(train_flooded_masks))

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

    return X_train, X_test, y_train, y_test

def model_preprocessing(X_train, X_test):
    preprocessor = PSPnet.preprocessing('resnet101')

    X_train = preprocessor(X_train)
    X_test = preprocessor(X_test)

    return X_train, X_test

def new_model(height=384, width=384):
    model = PSPnet.get_model(height,width)

    return model

# %%
# Global Variables

# Model specifications
seed = 42
np.random.seed = seed
num_imgs = 12
epochs = 25
train_grid_size = 768       # 384 or 768
model_name = 'PSPNet_'+str(num_imgs)+'img_'+str(epochs)+'epoch_'+str(train_grid_size)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")         

# General Prepocessing specifications
wd = os.getcwd()                # working directory
train_dir = wd+"/data/Train"    # training directory
n_classes=10                    # number of classes
height = train_grid_size
width = train_grid_size
img_height = 3000
img_width = 4000

h_n = int(img_height/height)
w_n = int(img_width/width)
# %%
X_train, X_test, y_train, y_test = general_preprocessing(train_dir, n_classes, height, width, img_height, img_width, h_n, w_n, seed, num_imgs)
X_train, X_test = model_preprocessing(X_train, X_test)
# %%
model = new_model(height, width)

# %%
# Training the model
tens_dir = wd+"/logs/tensorboard/"
csv_logfile = wd+"/logs/csvlogs/"+model_name+".csv"
model_path = wd+"/saved_models/"+model_name+"_{epoch:02d}-{val_iou_score:.2f}.hdf5"
callbacks = [
    keras.callbacks.TensorBoard(log_dir=tens_dir+model_name, histogram_freq=1),
    keras.callbacks.EarlyStopping(patience=3,monitor="val_iou_score", verbose=1),
    keras.callbacks.CSVLogger(csv_logfile),
    keras.callbacks.ModelCheckpoint(model_path, monitor="val_iou_score", verbose=1, save_best_only=True, mode='max')
    ]

history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
            )
# %%
model.save('saved_models/'+model_name+'final.hdf5')

# %%

#Plotting the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('img/graphs/'+model_name+'_loss.jpg')
plt.show()
# fig1 = plt.gcf()
# plt.show()
# plt.draw()
# fig1.savefig('img/graphs/'+model_name+'_loss.png')
# %%
acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

# %%
plt.plot(epochs, acc, 'b', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.savefig('img/graphs/'+model_name+'_IOU.jpg')
plt.show()
# fig2 = plt.gcf()
# plt.show()
# plt.draw()
# fig2.savefig('img/graphs/'+model_name+'_IOU.jpg')

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

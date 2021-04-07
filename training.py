# %%
import os
import numpy as np
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam

import PSPnet
# %%
wd = os.getcwd()                # working directory
train_dir = wd+"/data/Train"    # training directory
n_classes=10                    # number of classes
height = 768
width = 768
img_height = 3000
img_width = 4000

h_n = int(img_height/height)
w_n = int(img_width/width)

# %%
train_flooded_imgs=[]
train_flooded_img_dir = train_dir+"/Labeled/Flooded/image"
imgs = os.listdir(train_flooded_img_dir)
imgs.sort()
imgs = imgs[5:8]             # To be removed, truncating the number of images loaded for tesing purposes

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

# %%
train_flooded_masks = []
train_flooded_mask_dir = train_dir+"/Labeled/Flooded/mask"
masks = os.listdir(train_flooded_mask_dir)
masks.sort()
masks = masks[5:8]           # To be removed, truncating the number of masks loaded for tesing purposes

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

# %%
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

# %%
# Expanding dimensions because libraries expect and RGB channel.
train_flooded_masks = np.expand_dims(train_flooded_masks, axis=3)
# %%
# categorical loss function or other deep learning classification loss functions expect the input to be one hot encoded
n_classes = len(np.unique(train_flooded_masks))     # this line can be removed when sure about the number of classes
flood_masks_cat = to_categorical(train_flooded_masks, num_classes=n_classes, dtype='uint8')
train_flooded_masks_cat = flood_masks_cat.reshape((train_flooded_masks.shape[0],train_flooded_masks.shape[1],train_flooded_masks.shape[2],n_classes))

# %%
# Test-Train split for training
X_train, X_test, y_train, y_test = train_test_split(train_flooded_imgs, train_flooded_masks_cat, test_size=0.2, random_state=42)

# %%
# Calling model from PSPNet.py
optim = Adam()

model = PSPnet.get_model(height,width,optim)

# %%
preprocessor = PSPnet.preprocessing('resnet101')

X_train = preprocessor(X_train)
X_test = preprocessor(X_test)

# %%
# Training the model
model.fit(
    X_train,
    y_train,
    epochs=25,
    verbose=1,
    validation_data=(X_test, y_test)
)
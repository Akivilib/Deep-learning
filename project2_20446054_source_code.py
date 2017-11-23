#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:53:17 2017

@author: ylinbq
"""

import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, GlobalMaxPooling2D,Convolution2D,Activation

flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
num_training_img = 2570  #2569  3119
num_val_img = 549
num_test_img = 551
batch_size = 64
dim = 224

# bulid base model
base_model = ResNet50(weights='imagenet',include_top = False, input_shape = (224,224,3))
x = base_model.output
x = Flatten()(x)
predictions = Dense(5, activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=predictions)

# read img
train_datagen = ImageDataGenerator()

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    directory='data/train/flower_photos',  
    target_size=(dim, dim),
    batch_size=batch_size,
    shuffle=False)

validation_generator = val_datagen.flow_from_directory(
    directory='data/val/flower_photos',
    target_size=(dim, dim),
    batch_size=batch_size,
    shuffle=False)


test_data = list()
df = pd.read_csv('./data/test.txt', header = None, delim_whitespace=True)
for files in df.iloc[:,0].values:
    src = './data/' + files
    img = load_img(src, target_size = (224,224))
    img_array = img_to_array(img)
    test_data.append(img_array)
X_test = np.array(test_data)

# fit model
model.compile(optimizer=optimizers.rmsprop(0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=100)

# predict model
preds = model.predict(X_test, batch_size = 32, verbose = 1)
print(preds)

results = np.argmax(preds, axis=1)
print(results)

# save results
test_path = './Project2_20446054.txt'
file = open(test_path, 'w')
for i in results:
    file.write(str(i)+'\n')
file.close()


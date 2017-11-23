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
num_training_img = 2570
num_val_img = 549
num_test_img = 551
batch_size = 32
dim = 224

# bulid base model
base_model = ResNet50(weights='imagenet',include_top = False)
x = base_model.output
x = GlobalMaxPooling2D()(x)
#x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=predictions)

# read img
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='constant')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/flower_photos',  
    target_size=(dim, dim),  # all images will be resized to dim * dim
    batch_size=batch_size,
    class_mode='categorical')  # since loss = categorical_crossentropy, our class labels should be categorical

validation_generator = test_datagen.flow_from_directory(
    'data/val/flower_photos',
    target_size=(dim, dim),
    batch_size=batch_size,
    class_mode='categorical')


test_data = list()
df = pd.read_csv('./data/test.txt', header = None, delim_whitespace=True)
for files in df.iloc[:,0].values:
    src = './data/' + files
    img = load_img(src, target_size = (224,224))
    img_array = img_to_array(img)
    img_array /= 255 
    test_data.append(img_array)
X_test = np.array(test_data)

# first step: train top layers
for layer in base_model.layers:
    layer.trainable = False
    
model.fit_generator(
    train_generator,
    steps_per_epoch=num_training_img // batch_size,  # steps =  num_images // batch_size = total num of complete passes
    epochs=2)
    validation_data=validation_generator,
    validation_steps=num_val_img // batch_size)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# second step: freeze bottom 172 layers, train top 2 layers
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=num_training_img // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=num_val_img // batch_size)

# prediction
preds = model.predict(X_test, batch_size = 32, verbose = 1)
print(preds)

# write txt
results = []
for i in range(0, len(preds)):
    new_array = list(preds[i])
    results.append(new_array.index(preds[i].max()))
test_path = './Project2_20446054.txt'
file = open(test_path, 'w')
for i in results:
    file.write(str(i)+'\n')
file.close()




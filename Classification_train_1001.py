
import numpy as np
import os
import time
from vgg16 import VGG16  #vgg16.py is in vgg16_weight folder
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, merge, Input
from keras.models import Model, Sequential

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

import tesserpy
import cv2


################################################################################################################################################################

# Define data path, transform image data to the keras format
PATH = '/gpfs/scratch/bz957'
data_path = PATH + '/cropped'

img_data_list=[]
labels = []
bboxes = [] 

t=time.time()
i = 0
for folder in os.listdir(data_path):
    if folder.startswith('.'):
        continue
    file_path = data_path + '/' +folder
    for f in os.listdir(file_path):
        if not f.startswith('.') and os.path.isfile(os.path.join(file_path, f)):
            filepath = os.path.join(file_path, f)   
            ori_img = cv2.imread(filepath)
    
            #crop up 20% part of image to train           
            ori_img_h, ori_img_w = ori_img.shape[0], ori_img.shape[1]
            cropped_img = ori_img[0:int(0.2*ori_img_h),:]

            #transfer cv2 format to keras format
            cropped_img = cv2.resize(cropped_img, (224, 224))
            cropped_img = cropped_img[...,::-1].astype(np.float32)
            img = np.expand_dims(cropped_img, axis=0)
            x = preprocess_input(img)

            #append img and corresponding label
            img_data_list.append(x)
            labels.append(folder)        
                
            i += 1
            if i % 100 == 0 :
                print('Processing time: %s' % (t - time.time()))
                print('{} images processed'.format(i))  
                
img_data = np.array(img_data_list)
img_data=np.rollaxis(img_data,1,0)
img_data=img_data[0]

#One hot encoding of the labels
label = LabelBinarizer()
label.fit(labels)
Y = label.transform(labels)

#save npy
np.save('img_data', img_data)
np.save('Y', Y)

##################################################################################################################################################################################
#load image and label npy files
img_data = np.load('/gpfs/scratch/bz957/img_data.npy')
Y = np.load('/gpfs/scratch/bz957/Y.npy')

#Shuffle the dataset 
x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset (train_test_split has some bugs)
split = int(0.8*len(x))
X_train = x[:split]
X_test = x[split:]
y_train = y[:split]
y_test = y[split:]

################################################################################################################################################################
#Train Keras Classification model
num_classes = 5
batch_size=64
epochs = 32

#import VGG
image_input = Input(shape=(224, 224, 3))
model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

#customize classification model
last_layer = model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

# freeze all the layers except the last 3 dense layers
for layer in custom_vgg_model.layers[:-3]:
	layer.trainable = False

custom_vgg_model.summary()

#optimaizer
adadelta = optimizers.Adadelta(lr=0.5, rho=0.95, epsilon=None, decay=0.0)
custom_vgg_model.compile(loss='categorical_crossentropy',optimizer=adadelta,metrics=['accuracy'])

t=time.time()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

###########################################################################################################################################
# save the classification model and weights
model_json = custom_vgg_model.to_json()
with open("keras_vgg_c_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
custom_vgg_model.save('keras_vgg_c_model.h5')
print("Saved model to disk")
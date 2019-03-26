
import numpy as np
import os
import cv2
from keras.models import load_model, model_from_json
from keras.applications.imagenet_utils import preprocess_input

####################################################################################################################################################################
# Accuracy of non_insurance cards
# Define data path of non_insurance cards
PATH = '/gpfs/scratch/bz957'
non_data_path = PATH + '/pre_classification'

# load trained classification model's json file and create model
json_file = open('/gpfs/scratch/bz957/keras_vgg_c_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_c = model_from_json(loaded_model_json)

# load weights into new model
loaded_model_c.load_weights("/gpfs/scratch/bz957/keras_vgg_c_model.h5")

# threshold of probility confidence, >= threshold, the card is insur_card; otherwise not.
threshold = 0.9999999
accuracy = 0
recall = 0

j = 0
batch_size = 32
for folder in os.listdir(non_data_path):
    if folder.startswith('.'):
        continue
    file_path = non_data_path + '/' +folder
    for f in os.listdir(file_path):
        if not f.startswith('.') and os.path.isfile(os.path.join(file_path, f)):
            j += 1
            filepath = os.path.join(file_path, f) 

            #crop up 20% part of card to train  
            ori_img = cv2.imread(filepath) 
            try:        
                ori_img_h, ori_img_w = ori_img.shape[0], ori_img.shape[1]
            except:
                continue
            cropped_img = ori_img[0:int(0.2*ori_img_h),:]

            #transfer cv2 format to keras format
            cropped_img = cv2.resize(cropped_img, (224, 224))
            cropped_img = cropped_img[...,::-1].astype(np.float32)
            img = np.expand_dims(cropped_img, axis=0)
            x = preprocess_input(img)
            img_data = np.array(x)

            #use model to get the highest softmax score
            label_pred = loaded_model_c.predict(img_data)
            highest_pro = max(label_pred[0])    

            if highest_pro <= threshold:
                accuracy+=1

            if j % batch_size == 0 :
                print('{} images processed'.format(j)) 
 
accuracy = accuracy/j
print('%.4f' % accuracy)

####################################################################################################################################################################
# Recall of insurance cards

recall_i = 0
recall_j = 0
PATH = '/gpfs/scratch/bz957'

# Define data path
data_path = PATH + '/cropped'
filelist_test = np.load('/gpfs/scratch/bz957/filelist_test.npy')

for folder in os.listdir(data_path):
    if folder.startswith('.'):
        continue
    file_path = data_path + '/' +folder
    for f in os.listdir(file_path):
        if not f.startswith('.') and os.path.isfile(os.path.join(file_path, f)):
            if f in filelist_test:
                recall_j += 1
                filepath = os.path.join(file_path, f) 

                #crop up 20% part of card to train  
                ori_img = cv2.imread(filepath)         
                ori_img_h, ori_img_w = ori_img.shape[0], ori_img.shape[1]
                cropped_img = ori_img[0:int(0.2*ori_img_h),:]

                #transfer cv2 format to keras format
                cropped_img = cv2.resize(cropped_img, (224, 224))
                cropped_img = cropped_img[...,::-1].astype(np.float32)
                img = np.expand_dims(cropped_img, axis=0)
                x = preprocess_input(img)
                img_data = np.array(x)

                #use model to get the highest softmax score
                label_pred = loaded_model_c.predict(img_data)
                highest_pro = max(label_pred[0])
      
                if highest_pro <= threshold:
                    recall_i += 1


                if recall_j % batch_size == 0 :
                    print('{} images processed'.format(recall_j))                
 
recall = recall_i/recall_j
print('%.4f' % recall)
















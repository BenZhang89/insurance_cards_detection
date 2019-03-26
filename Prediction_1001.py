
import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers
import pandas as pd
import tesserpy
import cv2
import math
import re
import PIL
from PIL import Image
from keras.models import load_model, model_from_json

#########################################################################################

i = 0
num_classes = 5
batch_size=64
epochs = 32

#########################################################################################
# load trained classification model
json_file = open('/gpfs/scratch/bz957/keras_vgg_c_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_c = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_c.load_weights("/gpfs/scratch/bz957/keras_vgg_c_model.h5")

#########################################################################################
# second approach to extract member ID:

PATH = '/gpfs/scratch/bz957'
# Define data path
data_path = PATH + '/cropped'


ori_img_list = []
img_data_list=[]
labels = []
bboxes = []
ID_rightpred = 0
ID_total = 0
df = pd.read_csv('/gpfs/scratch/bz957/file2_1500*21samples.csv')
j = 0 
recall_dic = {}

for folder in os.listdir(data_path):
    if folder.startswith('.'):
        continue
    file_path = data_path + '/' +folder
    for f in os.listdir(file_path):
        if not f.startswith('.') and os.path.isfile(os.path.join(file_path, f)):
            docid = f.split('.')
            docid = docid[0]
            row = df.loc[(df['DOCUMENT_ID'] == docid)]

            #get the subscribe member id from metadata
            try:
                subscr_id = row.iloc[0]['SUBSCR_NUM']
            except:
                continue
                
            filepath = os.path.join(file_path, f) 

            #crop up 20% of insurance card as input of the prediction model
            ori_img = cv2.imread(filepath)         
            ori_img_h, ori_img_w = ori_img.shape[0], ori_img.shape[1]
            cropped_img = ori_img[0:int(0.2*ori_img_h),:]

            #transfer cv2 format to keras format
            cropped_img = cv2.resize(cropped_img, (224, 224))
            cropped_img = cropped_img[...,::-1].astype(np.float32)
            img = np.expand_dims(cropped_img, axis=0)
            x = preprocess_input(img)
            img_data = np.array(x)

            #find the predicted class  
            classes_ = np.array(['AETNA', 'AFFINITY', 'HEALTHFIRST', 'HIP', 'LOCAL 1199'],dtype='<U11') #if more insurs types, need to build idx_label.txt
            label_pred = loaded_model_c.predict(img_data)
            label_pred = classes_[np.argmax(label_pred)]

            # define which font to use according the insurance company prediction result
            if label_pred in ['AETNA', 'HIP','HEALTHFIRST']:
                label_font = label_pred
            else:
                label_font = 'eng'

            #scall originial image to 1500 width
            basewidth = 1500
            pil_image = Image.open(filepath).convert('RGB') 
            wpercent = (basewidth / float(pil_image.size[0]))
            hsize = int((float(pil_image.size[1]) * float(wpercent)))
            pil_image = pil_image.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
            # pil_image = PIL.Image.open('Image.jpg').convert('RGB') 
            open_cv_image = np.array(pil_image) 
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 

            # get the 0.1-0.8* height and up left 0.55 part of image for ocr
            open_cv_image_h, open_cv_image_w = open_cv_image.shape[0], open_cv_image.shape[1]
            cropped_img = open_cv_image[int(0.1*open_cv_image_h):int(0.8*open_cv_image_h), 0: int(0.55*open_cv_image_w)]

            # loat tesserpy and characters we need to use for member ID
            tess = tesserpy.Tesseract("/gpfs/home/bz957/.conda/pkgs/tesseract-3.05.02-h1ccaaf6_0/share/tessdata", language=label_font)
            tess.tessedit_char_whitelist = """#:.-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"""
            image_ocr = cropped_img.copy()

            #image preprocessing with OpenCV, to let ocr be more accurate on extracting the information from cards
            image_ocr = cv2.resize(image_ocr, (0,0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(image_ocr, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((1, 1), np.uint8)  # Apply dilation and erosion to remove some noise
            img_gray = cv2.dilate(img_gray, kernel, iterations=1)
            img_gray = cv2.erode(img_gray, kernel, iterations=1)
            img_gray = cv2.medianBlur(img_gray,3)
            ret,img_th = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)


            # use different 'check key' to identify the member ID; different insurance cards have different keys
            tess.set_image(img_th)
            tess.get_utf8_text()
            lastword = "none"
            predict_id = '0'
            longest_int = '0'
            id_list = []
            if label_pred == 'AETNA':
                #'check key'  for 'AETNA'
                checkinglist = ['Number)', 'Number:', 'ID:', 'ID#', 'ID#:','ID']
                for word in tess.words():  
                    #Member ID of AETNA is seperate by space, so need to combine two extracted numbers
                    combined = lastword + word.text
                    for check_word in checkinglist:
                        if check_word in word.text and len(word.text)>=8:
                            predict_id = word.text.split(check_word, maxsplit=1)[1] 
                            break
                        if check_word in lastword and len(word.text)>=8:
                            predict_id = word.text       
                            break

                    if len(predict_id)>= 7:
                        break

                    if len(word.text) > len(longest_int) and word.text.isdigit():
                        longest_int = word.text

                    if len(id_list)>0:
                        id_list.append(word.text)
                        predict_id = ''.join(id_list)
                        if len(predict_id)>= 7:
                            break    

                    if lastword in checkinglist:
                        id_list.append(word.text)
                        if len(id_list[0]) >= 7:
                            predict_id = id_list[0]
                            break

                    try:
                        if combined[0] == 'W' and len(combined[1:])>=7 and len(word.text)==5:
                            predict_id = combined
                            break

                    except:
                        continue

                    lastword = word.text
                if len(predict_id) <= 1:
                    predict_id = longest_int
                if predict_id[0] == 'W': 
                    #After 'W', all ID is number, not character. best approach is to retrain font, need to do it later
                    predict_id = predict_id.replace('S','5')
                    predict_id = predict_id.replace('E','8')

            if label_pred == 'AFFINITY':
                checkinglist = ['ID:','#']

                for word in tess.words():  
                    combined = lastword + word.text
                    for check_word in checkinglist:
                        if check_word in word.text and len(word.text)>=8:
                            predict_id = word.text.split(check_word, maxsplit=1)[1] 
                            break
                        if check_word in lastword and len(word.text)>=8:
                            predict_id = word.text       
                            break
                    if len(predict_id)>= 7:
                        break

                    if len(word.text) > len(longest_int) and word.text.isdigit():
                        longest_int = word.text

                    if len(id_list)>0:
                        id_list.append(word.text)
                        predict_id = ''.join(id_list)
                        if len(predict_id)>= 7:
                            break    
                    if lastword in checkinglist:
                        id_list.append(word.text)
                        if len(id_list[0]) >= 7:
                            predict_id = id_list[0]
                            break
               
                    lastword = word.text

                if len(predict_id) <= 1:
                    predict_id = longest_int


            if label_pred == 'HEALTHFIRST':
                checkinglist = ['ID:', '#']
                for word in tess.words():  
                    combined = lastword + word.text

                    for check_word in checkinglist:
                        if check_word in word.text and len(word.text)>=8:
                            predict_id = word.text.split(check_word, maxsplit=1)[1] 
                            break
                        if check_word in lastword and len(word.text)>=8:
                            predict_id = word.text       
                            break

                    if len(predict_id)>= 7:
                        break

                    if len(word.text) > len(longest_int) and word.text.isdigit():
                        longest_int = word.text                   

                    if len(id_list)>0:
                        id_list.append(word.text)
                        predict_id = ''.join(id_list)
                        if len(predict_id)>= 7:
                            break    

                    if lastword in checkinglist:
                        id_list.append(word.text)
                        if len(id_list[0]) >= 7:
                            predict_id = id_list[0]
                            break
      
                    lastword = word.text

                if len(predict_id) <= 1:
                    predict_id = longest_int

                if predict_id.replace('O','0').isdigit(): 
                    predict_id = predict_id.replace('O','0')

            if label_pred == 'HIP':
                checkinglist = ['NUMBER:','NUMBER','Member#:' , 'ID:', '#:','Number:','#']
                for word in tess.words():  
                    combined = lastword + word.text
                    for check_word in checkinglist:
                        if check_word in word.text and len(word.text)>=8:
                            predict_id = word.text.split(check_word, maxsplit=1)[1] 
                            break

                        if check_word in lastword and len(word.text)>=8:
                            predict_id = word.text       
                            break

                    if len(predict_id)>= 7:
                        break

                    if len(word.text) > len(longest_int) and word.text.isdigit():
                        longest_int = word.text

                    if len(id_list)>0:
                        id_list.append(word.text)
                        predict_id = ''.join(id_list)
                        if len(predict_id)>= 7:
                            break    

                    if lastword in checkinglist:
                        id_list.append(word.text)
                        if len(id_list[0]) >= 7:
                            predict_id = id_list[0]
                            break
      
                    lastword = word.text

                if len(predict_id) <= 1:
                    predict_id = longest_int


            if label_pred == 'LOCAL 1199':
                checkinglist = ['No.', 'ID:', 'ID', 'No']
                for word in tess.words():  
                    combined = lastword + word.text
                    for check_word in checkinglist:
                        if check_word in word.text and len(word.text)>=8:
                            predict_id = word.text.split(check_word, maxsplit=1)[1] 
                            break
                        if check_word in lastword and len(word.text)>=8:
                            predict_id = word.text       
                            break

                    if len(predict_id)>= 7:
                        break

                    if len(word.text) > len(longest_int) and word.text.isdigit():
                        longest_int = word.text

                    if lastword in checkinglist:
                        if len(word.text) >= 7:
                            predict_id = word.text
                            break

                    lastword = word.text

                if len(predict_id) <= 1:
                    predict_id = longest_int

            # remove special characters in the predicted ID
            # re.sub(predict_id, r'( "#:.-"))
            special_char = "#:.-"
            for char in special_char:
                predict_id = predict_id.replace(char,"")


            # AFFINITY has mislabelleb in metadata. This is just for count accuracy
            if label_pred == 'AFFINITY':
                temp1 = predict_id[:-2]
                temp2 = predict_id[1:]
                if subscr_id == temp1 or  subscr_id == temp2:
                    predict_id = subscr_id

            # compare the predicted ID and groundtruth ID
            ID_total += 1
            if predict_id == subscr_id:
                ID_rightpred += 1

            if predict_id != subscr_id:
                if predict_id == None:
                    predict_id = 'Can not be detected'
                recall_dic[docid] = subscr_id + '/' + predict_id

            j += 1
            if j % batch_size == 0 :
                print('{} images processed'.format(j)) 

accuracy =  ID_rightpred/ID_total

np.save('recall_dic_HEALTHFIRST.npy', recall_dic) 

# Load
print('%.2f' % accuracy)

# #########################################################################################

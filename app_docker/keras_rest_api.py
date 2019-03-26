# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import PIL
from PIL import Image
import numpy as np
import flask
from flask import Flask, render_template, request
import io
from keras.models import load_model, model_from_json
import cv2
from keras.applications.imagenet_utils import preprocess_input
import tesserpy


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    
    # load trained classification model's json file and create model
    json_file = open('./classification_model_weight/keras_vgg_c_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights("./classification_model_weight/keras_vgg_c_model.h5")
    #this is special for docker, without this will raise '<tensor> is not an element of this graph' error
    model._make_predict_function()

def prepare_image(image, target):
    # crop up 20% part of card to train 
    ori_img_h, ori_img_w = image.shape[0], image.shape[1]
    cropped_img = image[0:int(0.2*ori_img_h),:]

    #transfer cv2 format to keras format
    cropped_img = cv2.resize(cropped_img, (224, 224))
    cropped_img = cropped_img[...,::-1].astype(np.float32)
    img = np.expand_dims(cropped_img, axis=0)
    x = preprocess_input(img)
    img_data = np.array(x)

    # return the processed image
    return img_data

def find_ID(label_pred, pil_image):
    # define which font to use according the insurance company prediction result
    if label_pred in ['AETNA', 'HIP','HEALTHFIRST']:
        label_font = label_pred
    else:
        label_font = 'eng'

    #scall originial image to 1500 width
    basewidth = 1500

    # if the image mode is not RGB, convert it
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

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
    tess = tesserpy.Tesseract("/home/tesseract/tessdata", language=label_font)
    #/usr/local/Cellar/tesseract/3.05.02/share/tessdata
    #tess = tesserpy.Tesseract("/gpfs/home/bz957/.conda/pkgs/tesseract-3.05.02-h1ccaaf6_0/share/tessdata", language=label_font)
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
    
    return predict_id

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route("/uploader", methods=["POST"])
def uploader():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    threshold = 1-1e-7
    # ensure an image was properly uploaded to our endpoint
    # if flask.request.method == "POST":
    #    if flask.request.files.get("image"):
            # read the image in OpenCV format
    if request.method == 'POST':
        image = request.files['file'].read()     
        #image = flask.request.files["image"].read()

        #read the image in PIL format, for id prediction
        image_id = Image.open(io.BytesIO(image))

        # convert the image to a NumPy array and then read it into OpenCV format
        image_label = np.asarray(bytearray(image), dtype="uint8")
        ori_img = cv2.imdecode(image_label, cv2.IMREAD_COLOR)
        # preprocess the image and prepare it for classification
        img_data = prepare_image(ori_img, target=(224, 224))

        # classify the input image and then initialize the list
        # of predictions to return to the client
        #use model to get the highest softmax score

        # label_pred = model._make_predict_function(img_data)
        label_pred = model.predict(img_data)
        highest_pro = max(label_pred[0])    

        #find the predicted class  
        classes_ = np.array(['AETNA', 'AFFINITY', 'HEALTHFIRST', 'HIP', 'LOCAL 1199'],dtype='<U11') 
        #if more insurs types, need to build idx_label.txt 
        if highest_pro <= threshold:
            data["Pre_classification"] = 'Not an insurance card'
        else:
            data["Pre_classification"] = 'It is an insurance card'
            data["Label"] = classes_[np.argmax(label_pred)]			
            data["ID"] = find_ID(data["Label"], image_id)

            data["success"] = True
	    

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    print("model loaded, starting app")
    app.run(host='0.0.0.0', port=5000, debug=True)#defaut port=5000

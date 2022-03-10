from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import cv2 
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import pickle
import joblib
import base64
from PIL import Image

## Function to make predictions
def crop_pred(model, image):
    
    classes = ["Jute","Maize","Rice","Sugarcane","Wheat"]
    
    # print("This is your image : \n")
    # print("##############################################################################################################")
    # plt.imshow(image)
    # plt.show()
    # print("##############################################################################################################")

    image = cv2.resize(image, (64,64))

    # scale values from range 0-255 to 0-1
    image = image.astype("float") / 255.0

    # Ready the image for input to the network
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)[0]
    
    # result = pd.DataFrame( columns = ["Crops","Confidence"])
    # result.loc[len(result.index)] = [classes[0], round(float(list(preds)[0]), 3)]
    # result.loc[len(result.index)] = [classes[1], round(float(list(preds)[1]), 3)]
    # result.loc[len(result.index)] = [classes[2], round(float(list(preds)[2]), 3)]
    # result.loc[len(result.index)] = [classes[3], round(float(list(preds)[3]), 3)]
    # result.loc[len(result.index)] = [classes[4], round(float(list(preds)[4]), 3)]
    result = dict()
    result[classes[0]] = round(float(list(preds)[0]), 3)
    result[classes[1]] = round(float(list(preds)[1]), 3)
    result[classes[2]] = round(float(list(preds)[2]), 3)
    result[classes[3]] = round(float(list(preds)[3]), 3)
    result[classes[4]] = round(float(list(preds)[4]), 3)
    
    return result


## Load the model
model_cnn = load_model("Crop_CNN.h5")


app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'

@app.route('/prediction', methods=['POST'])
def predict_crop():
    
    # RECIEVE THE REQUEST 
    crop_image = request.files['image']
    img = Image.open(crop_image)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # image_dict = dict()
    # image_dict["cp_image"] = str(type(crop_image))
    # jpg_original = base64.b64decode(img_str)
    # jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    # crop_image = cv2.imdecode(jpg_as_np, flags=1)
    
    # PRINT THE DATA PRESENT IN THE REQUEST 
    print("[INFO] Request: ", img)
    
    # # PREDICT THE CLASS USING HELPER FUNCTION 
    results = crop_pred(model = model_cnn, image = img)
    
    # # PRINT THE RESULT 
    print("[INFO] Response: ", results)
          
    # # SEND THE RESULT AS JSON OBJECT 
    return jsonify(results)
    # return jsonify(image_dict)

if __name__ == '__main__':
    app.run()


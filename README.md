# Crops_Image_Classifcation
We will develop a a Crops Image Classifier based on Artificial Neural Networks and Convolutional Neural Networks, compare them and select the best and test the classifier.
Worldwide there are varieties of crops cultivated over ages. Each variety has its own distinguishing features over a broad spectrum of height, color, structure, number of seeds, etc. In taxonomy, there are various species, types of crops. The crops can be classified in various ways. Appearance is the first thing that will help in classification. Crops need to be identified as each crop needs specific treatment. We will try to use Artificial Neural Networks(ANN) to identify crops from images of different crops.

## About the dataset :
Dataset (Crop Images) contain 40+ images of each Agriculture crop(Maize, Wheat, jute, rice and sugarcane)
Dataset (kag2) contains 159+ augmented images of Crop Images of each class. Augmentation contain Horizontal flip, roatation, horizontal shift, vertical shift.
Test_data found at https://www.kaggle.com/aman2000jaiswal/testssss
https://www.kaggle.com/aman2000jaiswal/agriculture-crop-images

## About the model :

### ANN Model:
 
 Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param    
=================================================================
dense_4 (Dense)              (None, 3072)              37751808  
_________________________________________________________________
dropout (Dropout)            (None, 3072)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 768)               2360064   
_________________________________________________________________
dense_6 (Dense)              (None, 5)                 3845      
=================================================================
Total params: 40,115,717
Trainable params: 40,115,717
Non-trainable params: 0

### CNN Model:

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 64, 64, 10)        760       
_________________________________________________________________
activation_4 (Activation)    (None, 64, 64, 10)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 32, 10)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 20)        5020      
_________________________________________________________________
activation_5 (Activation)    (None, 32, 32, 20)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 20)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 5120)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 300)               1536300   
_________________________________________________________________
activation_6 (Activation)    (None, 300)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 1505      
_________________________________________________________________
activation_7 (Activation)    (None, 5)                 0         
=================================================================
Total params: 1,543,585
Trainable params: 1,543,585
Non-trainable params: 0
_____________________________

### Training and Validation Accuracy and Loss comparison.

![image](https://user-images.githubusercontent.com/84405967/157668374-b0797629-2fed-4919-936c-8de6f1f7d694.png)

We can see that graphs of CNN Model are smoother than ANN Model, thus there are no abrupt changes in the accuracy and loss. Similarly the CNN model has achieved better balance in training and validation parameters in less number of epochs.

### Few Sample Predictions:
![image](https://user-images.githubusercontent.com/84405967/157668482-0607f711-eaf8-48fa-b02e-fe95016822b8.png)

We can see that out of 16 samples, two samples have been missclassified by ANN Model and only one by CNN Model. We can say CNN Model is performing better job.
The accuracy of the ANN model : 85.57 % 
The accuracy of the CNN model : 91.04 % 
CNN model has 6 % more accuracy than the ANN model.

## Local Deployment
We will deploy the saved model.
Flask was used to create application and templates are developed in html.
Download all the contents of the Crop_Classifier_Deploy Folder.
Then in the address bar of the folder , type cmd and launch the command prompt.
Create new environment of preferred name by command - create conda -n name python=3.7
Then activate the environment - conda activate name 
Then we will install all required packages - pip install -r requirements.txt
Once everything is installed, we will run the app - python Local_Crop_App.py
Then the app runs on the local server at a given address, we can copy paste it in browser.

There is a also a Local_Deploy.ipynb for Deployment with API using Postman.

Thank You.



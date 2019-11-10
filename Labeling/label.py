import keras
import pandas as pd
import numpy as np
import shutil
from tensorflow.keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import os
import time


#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')

#"Data/Test/" + imageselect

directory = "C:/Users/Luke/Desktop/AI-Photography/Images/SampleImages/"
label_directory = "C:/Users/Luke/Desktop/AI-Photography/Images/SortedImages/"
testfile = "C:/Users/Luke/Desktop/AI-Photography/Images/SampleImages/DSC_0046.JPG"
def predict(model1, file):
    img_width, img_height = 224,224
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model1.predict(x)
    return array

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".JPG"):
         start = time.time()
         prediction = predict(vgg_model, directory+filename)
         end = time.time()
         print("Inference took: " + str(round(end - start, 2)) + " seconds.")
         decoded = decode_predictions(prediction, top=1)[0]
         print('Predicted:', decoded)
         category = decoded[0][1]
         if not os.path.exists(label_directory+category):
             os.makedirs(label_directory+category)
             shutil.copy(directory+filename, label_directory+category)
         if os.path.exists(label_directory+category):
             shutil.copy(directory+filename, label_directory+category)
     else:
         continue

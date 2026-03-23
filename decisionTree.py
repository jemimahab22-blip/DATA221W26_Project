import numpy as np
import os
import cv2
import kagglehub
from sklearn.tree import DecisionTreeClassifier

#This dataset was already split, trained, validated and tested
#We now have to convert the images into features since the Decision Tree Classifier does not work well with images according the research.
#Decision trees need tabular data not images.

def load_data(path, label):
    data =[]
    labels =[]

    for image_name in os.listdir(path):  #I am iterating through the train path to find the image names
        image_path= os.path.join(path, image_name)

        image = cv2.imread(image_path)   #I am reading the image_path
        if image is None:
            continue  #this skips bad images

        image = cv2.resize(image, (100,100))   #resizing the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting to the grayscale

        image = image.flatten() #Flattening the image to 1D

        data.append(image)
        labels.append(label)

    return data, labels

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)

#I am digging into the path to access the exact data-set to be used
data_path = os.path.join(path,"chest_xray")

X_train, y_train =[],[]
X_validation, y_validation=[],[]
X_test, y_test=[],[]

#I am training the model
data, labels = load_data(os.path.join(data_path ,"train","NORMAL"),0)
X_train+= data ; y_train+=labels

data, labels = load_data(os.path.join(data_path,"train","PNEUMONIA"),1)
X_train+=data ; y_train+=labels

#Validation
data, labels = load_data(os.path.join(data_path,"val","NORMAL"),0)
X_validation+=data ; y_validation+=labels

data, labels =load_data(os.path.join(data_path,"val","PNEUMONIA"),1)
X_validation+=data ; y_validation+=labels

#testing the model
data, labels = load_data(os.path.join(data_path, "test","NORMAL"),0)
X_test+=data ; y_test+=labels

data,labels = load_data(os.path.join(data_path, "test","PNEUMONIA"),1)
X_test+=data ; y_test+=labels
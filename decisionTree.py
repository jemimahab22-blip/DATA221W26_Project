import numpy as np
import os
import cv2
from sklearn.tree import DecisionTreeClassifier

#This dataset was already split, trained, validated and tested
#We now have to convert the images into features since the Decision Tree Classifier does not work well with images according the research.
#Decision trees need tabular data not images.

#here, I am getting the dataset from the path
dataset_path = "C:/Users/jemim/OneDrive - University of Calgary/Desktop/DATA 221/Pneumonia-Chest Xray/chest_xray"

train_path= os.path.join(dataset_path)  #I am getting the train path using the dataset_path

#print(os.listdir(train_path))  #this prints out what is contained in the train_path

def load_data(dataset_path, label):
    data =[]
    labels =[]

    for image_name in train_path:  #I am iterating through the train path to find the image names
        image_path= os.path.join(dataset_path, image_name)

        image = cv2.imread(image_path)   #I am reading the image_path
        image = cv2.resize(image, (200,200))   #resizing the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting to the grayscale

        image = image.flatten() #Flattening the image to 1D

        data.append(image)
        labels.append(label)

    return data, labels





import numpy as np
import os
import cv2
from sklearn.tree import DecisionTreeClassifier

#This dataset was already split, trained, validated and tested
#We now have to convert the images into features since the Decision Tree Classifier does not work well with images according the research.
#Decision trees need tabular data not images.

def load_data(folder_path, label):
    data =[]
    labels =[]

    for image_name in os.listdir(folder_path):  #I am iterating through the train path to find the image names
        image_path= os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)   #I am reading the image_path
        if image is None:
            continue  #this skips bad images

        image = cv2.resize(image, (100,100))   #resizing the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting to the grayscale

        image = image.flatten() #Flattening the image to 1D

        data.append(image)
        labels.append(label)

    return data, labels
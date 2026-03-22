import numpy as np
import os
import cv2

#This dataset was already split, trained, validated and tested
#We now have to convert the images into features since the Decision Tree Classifier does not work well with images according the research.
#Decision trees need tabular data not images.

#here, I am getting the dataset from the path
dataset_path = "C:/Users/jemim/OneDrive - University of Calgary/Desktop/DATA 221/Pneumonia-Chest Xray/chest_xray"

train_path= os.path.join(dataset_path, "train")  #I am getting the train path using the dataset_path

print(os.listdir(train_path))  #this prints out what is contained in the train_path



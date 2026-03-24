import numpy as np
import os
import cv2
import kagglehub
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#This dataset was already split, trained, validated and tested
#We now have to convert the images into features since the Decision Tree Classifier does not work well with images according the research.
#Decision trees need tabular data not images.

def load_data(path, label):
    data_ =[]
    labels_ =[]

    for image_name in os.listdir(path):  #I am iterating through the train path to find the image names
        image_path= os.path.join(path, image_name)

        image = cv2.imread(image_path)   #I am reading the image_path
        if image is None:
            continue  #this skips bad images

        image = cv2.resize(image, (100,100))   #resizing the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting to the grayscale

        image = image.flatten() #Flattening the image to 1D

        data_.append(image)
        labels_.append(label)

    return data_, labels_

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("\nPath to dataset files:", path)
print(os.listdir(path))

#I am digging into the path to access the exact data-set to be used
data_path = os.path.join(path,"chest_xray")

X_train, y_train =[],[]
X_validation, y_validation=[],[]
X_test, y_test=[],[]

#I am training the model
data_, labels_ = load_data(os.path.join(data_path ,"train","NORMAL"),0)
X_train+= data_ ; y_train+=labels_

data_, labels_ = load_data(os.path.join(data_path,"train","PNEUMONIA"),1)
X_train+=data_ ; y_train+=labels_

#Validation
data_, labels_ = load_data(os.path.join(data_path,"val","NORMAL"),0)
X_validation+=data_ ; y_validation+=labels_

data_, labels_ =load_data(os.path.join(data_path,"val","PNEUMONIA"),1)
X_validation+=data_ ; y_validation+=labels_

#testing the model
data_, labels_ = load_data(os.path.join(data_path, "test","NORMAL"),0)
X_test+=data_ ; y_test+=labels_

data_,labels_ = load_data(os.path.join(data_path, "test","PNEUMONIA"),1)
X_test+=data_ ; y_test+=labels_

#using arrays instead of lists
X_train= np.array(X_train)
y_train= np.array(y_train)

X_validation=np.array(X_validation)
y_validation= np.array(y_validation)

X_test= np.array(X_test)
y_test= np.array(y_test)

#model fitting
model_fitted = DecisionTreeClassifier(min_impurity_decrease=0.01,
                                      min_samples_split=2,
                                      criterion='entropy',
                                      max_depth=5,
                                      random_state=42)
model_fitted.fit(X_train,y_train)

#predicting the model
model_predicted=model_fitted.predict(X_test)

#predicting the validation of the model
validation_predicted= model_fitted.predict(X_validation)


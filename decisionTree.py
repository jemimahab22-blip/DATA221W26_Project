import numpy as np
import os
import cv2
import kagglehub
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay
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
        image = image/225.0 #Normalizing the image

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

print("\nTrain NORMAL:", np.sum(y_train == 0))
print("\nTrain PNEUMONIA:", np.sum(y_train == 1))

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

#getting the metrics from the dataset
#accuracies
accuracy1 = accuracy_score(y_test,model_predicted)
accuracy2 = accuracy_score(y_validation,validation_predicted)
#precisions
precision1= precision_score(y_test,model_predicted)
precision2 = precision_score(y_validation,validation_predicted)
#recall
recall1= recall_score(y_test,model_predicted)
recall2 = recall_score(y_validation,validation_predicted)
#f1_scores
f1_1 = f1_score(y_test,model_predicted)
f1_2 = f1_score(y_validation,validation_predicted)
#ROC_AUC
y_probs = model_fitted.predict_proba(X_test)[:,1]
y_probs_= model_fitted.predict_proba(y_validation)[:,1]
roc_auc1 = roc_auc_score(y_test, y_probs)
roc_auc2 = roc_auc_score(y_validation,y_probs_)

print("\nROC-AUC1:", roc_auc1 ,"and ROC-AUC2:", roc_auc2)
print("\nThe accuracy of the tree is: ",accuracy1,"and the validation accuracy is: ",accuracy2)
print("\nThe precision1 score is: ",precision1, "and the precision2 score is: ",precision2)
print("\nThe recall score1 is: ",recall1, "and the recall score2 is:",recall2)
print("\nThe f1 score1 is: ",f1_1, "and the f1 score2 is: ",f1_2)


import numpy as np
import os
import cv2
import kagglehub
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,confusion_matrix
import matplotlib.pyplot as plt

# All models use the same preprocessing pipeline (resize, grayscale, normalization)
# to ensure a fair comparison across models

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
        image = image/255.0 #Normalizing the image

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
y_probs_= model_fitted.predict_proba(X_validation)[:,1]
roc_auc1 = roc_auc_score(y_test, y_probs)
roc_auc2 = roc_auc_score(y_validation,y_probs_)

print("\n--- Decision Tree Performance ---")
print(f"Test Accuracy: {accuracy1:.4f}")
print(f"Validation Accuracy: {accuracy2:.4f}")
print(f"Precision: {precision1:.4f}")
print(f"Recall: {recall1:.4f}")
print(f"F1 Score: {f1_1:.4f}")
print(f"ROC-AUC: {roc_auc1:.4f}")

# Interpretation:
# High recall is important in this problem because missing pneumonia cases (false negatives)
# can be dangerous in a medical context.
# Precision shows how reliable positive predictions are.

#getting the confusion matrix between the dataset
cm = confusion_matrix(y_test,model_predicted)
print("\nThis is the confusion matrix for the training dataset: ",cm)

import seaborn as sns

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=["NORMAL", "PNEUMONIA"],
            yticklabels=["NORMAL", "PNEUMONIA"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

tn, fp, fn, tp = cm.ravel() #this flattens each confusion matrix into a 1D and assigns each value

print(f"True Negatives (Correct NORMAL): {tn}")
print(f"False Positives (Wrongly predicted PNEUMONIA): {fp}")
print(f"False Negatives (Missed PNEUMONIA): {fn}")
print(f"True Positives (Correct PNEUMONIA): {tp}")

# In medical diagnosis; False Negatives (FN) are the most critical error because they represent missed pneumonia cases.

import numpy as np
import os
import cv2
import kagglehub
from numpy.ma.core import reshape
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
print(f"F1 Score(Test): {f1_1:.4f}")
print(f"ROC-AUC(Test): {roc_auc1:.4f}")

"""
# Interpretation:
# High recall is important in this problem because missing pneumonia cases (false negatives) 
#can be dangerous in a medical context.
# Precision shows how reliable positive predictions are.
"""

# adding the cross-validation-score
from sklearn.model_selection import cross_val_score

cv_f1 = cross_val_score(model_fitted, X_train, y_train, cv=5, scoring='f1')
cv_auc = cross_val_score(model_fitted, X_train, y_train, cv=5, scoring='roc_auc')

print("\n--- Cross-Validation Stability ---")
print(f"F1 Score CV Mean: {cv_f1.mean():.4f}")
print(f"F1 Score CV Std: {cv_f1.std():.4f}")
print(f"ROC-AUC CV Mean: {cv_auc.mean():.4f}")
print(f"ROC-AUC CV Std: {cv_auc.std():.4f}")
"""
Interpretation:
-Low standard deviation indicates the model performs consistently across the folds,
-meaning it is stable and not overly sensitive to the training data.
"""

# getting the most efficient feature in the decision tree
importance = model_fitted.feature_importances_

importance_image = importance.reshape(100,100)

plt.imshow(importance_image,cmap='hot')
plt.colorbar()
plt.title("Decision Tree most influential feature.")
plt.ylabel("Pixel y position (Length)")
plt.xlabel("Pixel X position (width)")
plt.axis('on')
plt.show()
# Interpretation:
#The bright regions represent the pixels that strongly influence classification.
#this provides insight into which the areas of the X-ray model relies on.


#np.where returns the indices of an array that satisfies the condition
misclassified_indices = np.where(model_predicted != y_test)[0]
plt.figure(figsize=(10,5)) #this is the figure size of the images

#I am looping through each index i in a range of 3
for i in range(min(3,len(misclassified_indices))):
    idx = misclassified_indices[i]
    labels = ["NORMAL", "PNEUMONIA"]
    plt.subplot(1,3,i+1)
    plt.imshow(X_test[idx].reshape(100,100), cmap='gray')
    plt.title(f"True: {labels[y_test[idx]]}\nPredicted: {labels[model_predicted[idx]]}")
    plt.suptitle("Misclassified X-ray Images (Decision Tree)", fontsize=14)
    plt.axis('off')

plt.show()

"""
Preprocessing is applied independently to each dataset split.
No information from validation/test is used during training (prevents data leakage).
The dataset is already split into train/val/test with class distribution preserved.

Limitation:
-Decision Trees are not well-suited for image data.
-Flattening images removes spatial relationships between pixels,
-which reduces the model's ability to detect meaningful patterns.

Future Improvement:
-Convolutional Neural Networks (CNNs) would perform better
-because they preserve spatial structure in images.

Success Criteria Evaluation met:

1. Performance (F1-score and ROC-AUC):
- Decision Tree F1 Score: {:.4f}
- Decision Tree ROC-AUC: {:.4f}

These values will be compared against the Logistic Regression baseline (provided separately) to determine if performance is improved.

2. Stability:
- Cross-validation shows mean F1 of {:.4f} with std {:.4f}
- Cross-validation shows mean ROC-AUC of {:.4f} with std {:.4f}

Low standard deviation indicates stable model performance.

3. Interpretability:
- Feature importance visualization highlights which pixels influence decisions.
- This is a key advantage of Decision Trees over Logistic Regression.

Conclusion:
- The Decision Tree (meets / does not meet) the success criteria depending on
  whether its F1 and ROC-AUC exceed the Logistic Regression baseline.
""".format(f1_1, roc_auc1, cv_f1.mean(), cv_f1.std(), cv_auc.mean(), cv_auc.std())
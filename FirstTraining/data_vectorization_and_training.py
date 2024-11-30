import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Step 1: Load the dataset
data = pd.read_csv("10k_samples_10_families_before_and_after_reboot.csv")
print("Dataset Loaded Successfully")
print("First 5 Rows of the Dataset:")
print(data.head())

# Step 2: Show information about the dataset
print("\nDataset Information:")
print(data.info())

# Step 3: Extract features and labels
X = data.drop(columns=["Hash", "Category", "Family"])
y = data["Family"]

# Show the extracted features and labels
print("\nFeatures (X) Preview:")
print(X.head())
print("\nLabels (y) Preview:")
print(y.head())

# Step 4: Vectorize the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("\nLabels After Encoding:")
print(y[:5])

# Step 5: Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("\nFeatures After Standardization (First 5 Rows):")
print(X[:5])

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nData Split into Training and Testing Sets")
print(f"Training Set Size: {X_train.shape[0]} samples")
print(f"Testing Set Size: {X_test.shape[0]} samples")

# Step 7: Initialize and train the Decision Tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
print("\nDecision Tree Classifier Trained Successfully")

# Step 8: Make predictions and evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 9: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 10: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot and save the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")  # Save the confusion matrix plot
plt.close()  # Close the plot to prevent displaying it

# Step 11: ROC Curve and AUC
# Compute ROC curve and AUC for each class (if multi-class, we use one-vs-rest approach)
y_score = classifier.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}

# Calculate ROC for each class
for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot and save the ROC curves
plt.figure(figsize=(10, 7))
for i in range(len(label_encoder.classes_)):
    plt.plot(fpr[i], tpr[i], label=f"Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Each Class")
plt.legend(loc="lower right")
plt.savefig("roc_curves.png")  # Save the ROC curve plot
plt.close()  # Close the plot to prevent displaying it

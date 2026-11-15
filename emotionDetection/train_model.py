import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = np.loadtxt("data.txt")  # Assuming the data is saved in a text file with features and labels

# Split features and labels
X = data[:, :-1]  # Everything except the last column
y = data[:, -1]   # Last column (labels)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print(confusion_matrix(y_test, predictions))

with open('./model', 'wb') as f:
    pickle.dump(model, f)



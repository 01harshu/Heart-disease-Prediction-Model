import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('C:/Users/GJPC/Desktop/cleaned_heart_data.csv')
X = df.drop('num', axis=1)
y = df['num']
X = pd.get_dummies(X)

# 2. Train and Predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 3. Print the Accuracy
accuracy = accuracy_score(y_test, predictions) * 100
print(f"Model Prediction Accuracy: {accuracy:.2f}%")

# 4. Show the Graph
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 1. Load the cleaned data
df = pd.read_csv('C:/Users/GJPC/Desktop/heart_disease_uci.csv')

# 2. Separate Features (X) and Target (y)
# Assuming 'num' is the column name for the disease status
X = df.drop('num', axis=1)
y = df['num']

# 3. Handle categorical data (converts text/categories into numbers)
X = pd.get_dummies(X)

# 4. Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 5. Extract and sort feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
top_10 = importances.nlargest(10)

# 6. Visualize the top 10 features
plt.figure(figsize=(10, 6))
top_10.sort_values().plot(kind='barh', color='teal')
plt.title("Top 10 Most Important Features for Heart Disease Prediction")
plt.xlabel("Relative Importance Score")
plt.ylabel("Medical Factors")
plt.tight_layout()
plt.show()
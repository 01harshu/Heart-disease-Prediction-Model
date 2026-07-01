
import pandas as pd

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\GJPC\Desktop\heart_disease_uci.csv')

# 2. Drop columns that are just labels or IDs and don't help in prediction
# Check your Excel view: if you see columns like 'id', 'name', 'dataset', add them here
columns_to_drop = ['id', 'name', 'dataset'] 
df = df.drop(columns=columns_to_drop, errors='ignore')

# 3. Fill missing values (NaN) with the median of each column
# This prevents the model from ignoring rows with empty data
df = df.fillna(df.median(numeric_only=True))

# 4. Save the cleaned data to use for your prediction model
df.to_csv(r'C:\Users\GJPC\Desktop\cleaned_heart_data.csv', index=False)

print("Cleaning complete! 'cleaned_heart_data.csv' has been saved to your Desktop.")
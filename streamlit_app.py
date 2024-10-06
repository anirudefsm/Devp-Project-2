
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample Data Loading Placeholder
# Replace 'sample.csv' with your actual data source or load from a database
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\aniru\Desktop\PythonProj\Imports_Exports_Dataset.csv")

data = load_data()

st.title('Visualization and Modeling Dashboard')

# 1. Bar Plot of Top 5 Countries by Average Value and Quantity
st.subheader('Top 5 Countries by Average Value and Quantity')

# Group by 'Country' and calculate the average for 'Value' and 'Quantity'
country_avg = data.groupby('Country').agg(
    Avg_Value=('Value', 'mean'),
    Avg_Quantity=('Quantity', 'mean')
).reset_index()

# Sort by 'Avg_Value' and 'Avg_Quantity' in descending order
country_avg_sorted = country_avg.sort_values(by=['Avg_Value', 'Avg_Quantity'], ascending=False)

# Get the top 5 countries with highest average value and quantity
top_5_countries = country_avg_sorted.head(5)
top_5_countries.index = top_5_countries.index + 1

# Create a side-by-side bar plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Country", y="Avg_Value", data=top_5_countries, color="blue", label="Avg_Value", ax=ax)
sns.barplot(x="Country", y="Avg_Quantity", data=top_5_countries, color="orange", label="Avg_Quantity", ax=ax)

# Set plot title and labels
ax.set_title("Top 5 Countries by Average Value and Quantity")
ax.set_ylabel("Value/Quantity")
ax.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
st.pyplot(fig)

# 2. Density Plot with KDE (Distribution of Value)
st.subheader('Distribution with Density Curve')

fig2, ax2 = plt.subplots()
sns.histplot(data['Value'], kde=True, color='blue', ax=ax2)
ax2.set_title('Distribution with Density Curve', fontsize=15)
ax2.set_xlabel('Data Values')
ax2.set_ylabel('Frequency')

st.pyplot(fig2)

# 3. Second-Degree Polynomial Regression to Predict Weight based on Quantity for Import Transactions
st.subheader('Polynomial Regression (Import Transactions)')

# Filter for Import transactions
df_imports = data[data['Import_Export'] == 'Import']

# Define dependent and independent variables
X = df_imports[['Quantity']]
y = df_imports['Weight']

# Transform the input for a polynomial regression (second degree)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Display model coefficients
st.write("Model Coefficients:", model.coef_)
st.write("Intercept:", model.intercept_)

# 4. Logistic Regression to Predict Import or Export Based on Value and Weight
st.subheader('Logistic Regression (Import/Export Prediction)')

# Define dependent and independent variables
X = data[['Value', 'Weight']]
y = data['Import_Export']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = logistic_model.predict(X_test)
st.text('Classification Report:')
st.text(classification_report(y_test, y_pred))

# End of Streamlit App

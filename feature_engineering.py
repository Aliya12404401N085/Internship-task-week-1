import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
df = pd.read_csv("titanic.csv")

# Check dataset
print(df.head())
print(df.info())

# Missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Feature Engineering
df["Fare_Per_Age"] = df["Fare"] / (df["Age"] + 1)

# Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard Scaling
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# MinMax Scaling
mm = MinMaxScaler()
X_train_mm = mm.fit_transform(X_train)
X_test_mm = mm.transform(X_test)

print("Preprocessing Completed Successfully")

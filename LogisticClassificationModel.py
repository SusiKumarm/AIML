import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(url)
print(df.head())
# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classification model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
scre=model.score(X_train,y_train)

print ("Training Score:", scre)


# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

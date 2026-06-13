import pandas as pd

df = pd.read_csv("dataset/spam.csv")

print(df.columns)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("dataset/spam.csv")

# Keep only required columns
df = df[['spamORham', 'Message']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].str.lower()

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Remove null values
df.dropna(inplace=True)

# Input and output
X = df['message']
y = df['label']

# Text vectorization
cv = CountVectorizer()

X = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

# Save model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

print("✅ Model Saved Successfully!")
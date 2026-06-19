# TASK SPAM DETECTION SYSTEM USING NAVIE BAYES AND LOGISTIC REGRESSION WITH VOICE INPUT AND OUTPUT BY PURNANKIT SIRVI
# TRAINING SCRIPT
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# LOAD DATASET
df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# CONVERSION FROM LABELS TO NUMBERS
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# VECTORIZATION
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# TRAIN NAIVE BAYES
model_nb = MultinomialNB()
model_nb.fit(X_train_vec, y_train)

# TRAIN LOGISTIC REGRESSION
model_lr = LogisticRegression()
model_lr.fit(X_train_vec, y_train)

# EVALUATION
pred_nb = model_nb.predict(X_test_vec)
pred_lr = model_lr.predict(X_test_vec)

print("Naive Bayes Accuracy:", accuracy_score(y_test, pred_nb))
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))

# SAVE MODELS AND VECTORIZER
pickle.dump(model_nb, open("model_nb.pkl", "wb"))
pickle.dump(model_lr, open("model_lr.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Models saved successfully!")
# TWITTER-SENTIMENT-ANALYSIS
Pandas, Scikit-learn, Streamlit

Here's a step-by-step walkthrough of the code:

1. Import the necessary libraries:
```python
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
```

2. Read the CSV file and create independent and dependent variables:
```python
Twitter_DF = pd.read_csv(r'https://raw.githubusercontent.com/gladinv/TWITTER-SENTIMENT-ANALYSIS/main/TWITTER_PROCESSED.csv')
Independent_var = Twitter_DF['text'] 
Dependent_var = Twitter_DF['final_target']
```

3. Split the data into training and testing sets:
```python
IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size=0.1, random_state=225)
```

4. Create a TfidfVectorizer object and a LogisticRegression object:
```python
tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver="lbfgs", max_iter=200000)
```

5. Create a pipeline with the vectorizer and classifier objects:
```python
model = Pipeline([('vectorizer',tvec),('classifier',clf2)])
```

6. Fit the model on the training set:
```python
model.fit(IV_train, DV_train)
```

7. Make predictions on the test set:
```python
predictions = model.predict(IV_test)
```

8. Create a Streamlit app:
```python
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon=':bird:', layout="centered")
st.title("Twitter Sentiment Analysis")
st.markdown("_***This Streamlit app is designed to analyze the sentiments of tweets collected from the social networking website Twitter.***_")
img = Image.open("imagestwitter.png")
st.image(img)
```

9. Create a text input for the user to enter a tweet:
```python
user_input = st.text_input("ENTER A TWEET: ")
```

10. If the user has entered some text, make a prediction and print the result:
```python
if user_input:
    result = model.predict([user_input])[0]
    sentiment = "Positive" if result == 2 else "Negative"
    st.write(f"### PREDICTED SENTIMENT: {sentiment}")

    # Print accuracy, precision, recall
    accuracy = accuracy_score(predictions, DV_test)
    precision = precision_score(predictions, DV_test, average='weighted')
    recall = recall_score(predictions, DV_test, average='weighted')
    f1 = f1_score(predictions, DV_test, average='weighted')
    st.write("### PERFORMANCE METRICS")
#     st.write(f"Accuracy: {accuracy:.2f}")
#     st.write(f"Precision: {precision:.2f}")
#     st.write(f"Recall: {recall:.2f}")
    st.write(f"### F1 SCORE: {f1:.2f}")
```

This code sets up a Streamlit app for sentiment analysis of tweets using a trained classification model. It allows users to input a tweet and get the predicted sentiment, along with the F1 score as a performance metric. The app also includes a title, a brief description, and an image related to Twitter sentiment analysis.

import streamlit as st
import re
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Define the create_feature function
def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

# Define the ngram function
def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

# Define the read_data function
def read_data(file):
    data = []
    with open(file, 'r')as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

# Define the convert_label function
def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()

# Load the data and preprocess it
file = 'data.txt'
data = read_data(file)
emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
x_all = []
y_all = []

for label, text in data:
    y_all.append(convert_label(label, emotions))
    x_all.append(create_feature(text, nrange=(1, 4)))

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 0.2, random_state = 123)

# Create a DictVectorizer to convert Counter objects to numeric feature vectors
vectorizer = DictVectorizer(sparse=True)

# Convert your feature data to numeric feature vectors
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Train the models
svc = SVC()
lsvc = LinearSVC()
rforest = RandomForestClassifier()
dtree = DecisionTreeClassifier()

svc.fit(x_train, y_train)
lsvc.fit(x_train, y_train)
rforest.fit(x_train, y_train)
dtree.fit(x_train, y_train)

# Define a function to predict emotion using the best model
def predict_emotion(text):
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    svc_accuracy = accuracy_score(y_test, svc.predict(x_test))
    lsvc_accuracy = accuracy_score(y_test, lsvc.predict(x_test))
    rforest_accuracy = accuracy_score(y_test, rforest.predict(x_test))
    dtree_accuracy = accuracy_score(y_test, dtree.predict(x_test))

    best_model = max([
        ("SVC", svc_accuracy),
        ("LinearSVC", lsvc_accuracy),
        ("RandomForest", rforest_accuracy),
        ("DecisionTree", dtree_accuracy)
    ], key=lambda x: x[1])

    model_name, _ = best_model
    
    # Map the model_name to the corresponding emoji
    emoji_dict = {
        "SVC": "😃",
        "LinearSVC": "😄",
        "RandomForest": "😁",
        "DecisionTree": "😆"
    }

    return emoji_dict.get(model_name, "Unknown")

# Define the Streamlit UI
st.title("Emotion Analyzer")

text_input = st.text_area("Enter your text:")
if st.button("Predict Emotion"):
    prediction = predict_emotion(text_input)
    st.write(f"Predicted Emotion: {prediction}")



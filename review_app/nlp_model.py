# NLP model to predict restaurant reviews

# Location of BASE_DIR
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(os.path.join(BASE_DIR, "review_app/nlp.sav"))

# Load the model
import pickle
classifier = pickle.load(open(os.path.join(BASE_DIR, 'review_app/nlp.sav'), 'rb'))

# Load the vectorizer
import pickle
cv = pickle.load(open(os.path.join(BASE_DIR, 'review_app/vectorizer.sav'), 'rb'))

# Import the libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def format_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

def predict(new_review):
    new_review = new_review
    new_review = format_review(new_review)
    corpus = []
    corpus.append(new_review)
    X_new_test = cv.transform(corpus).toarray()
    return classifier.predict(X_new_test)

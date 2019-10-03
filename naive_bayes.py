import utility
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

email_path = 'spam_classifier\\trec07p\\trec07p\data\inmail.'
label_path = 'spam_classifier\\trec07p\\trec07p\\full\index'
count_vect = CountVectorizer()
emails = []
raw_counts = [] # raw count of words in emails
labels = []
for i in range(1,301):
    emails.append(' '.join(utility.read_email(email_path+str(i))))

raw_counts = count_vect.fit_transform(emails)
transformer = TfidfTransformer().fit(raw_counts)
tf_idf = transformer.transform(raw_counts)

labels = utility.read_labels(label_path)[:300]


X_train, X_test, y_train, y_test = train_test_split(tf_idf, labels, test_size=0.3, random_state=69)
model = MultinomialNB().fit(X_train, y_train)
predicted = model.predict(X_test)
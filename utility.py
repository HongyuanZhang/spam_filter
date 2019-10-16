# to parse email
import email
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# machine learning tools
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import os

# for lemmatizing and getting rid of stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Input: a string of email contents
# Output: a list of lemmatized, meaningful words
def clean_up_plain_text(string):
    words = list()
    # split string into words
    tokens_raw = nltk.word_tokenize(string)
    i = 0
    # loop over raw tokens
    while i < len(tokens_raw):
        # current token
        cur = tokens_raw[i]
        # if current token is a potentially valid, meaningful word
        if cur.isalpha() and len(cur) > 1:
            # perform lemmatization on the lowercase of the token
            word = lemmatizer.lemmatize(cur.lower())
            # check if the token, after lemmatization, is an unmeaningful word
            if word not in stop_words:
                # if meaningful, append to the word list
                words.append(word)
        i += 1
        continue
    return words


# Input: a string of html contents
# Output: a list of meaningful words
def clean_up_html(html):
    words=list()
    # split string into words
    tokens_raw = nltk.word_tokenize(html)
    inside_tag = False
    i = 0
    # go over the tokens
    while i < len(tokens_raw):
        cur = tokens_raw[i]  # current token
        # if encountering '<', which starts a tag of html
        if cur == '<' and not inside_tag:
            inside_tag = True
            i += 1
            continue
        if cur == '>' and inside_tag:  # if reaches the end of an html tag
            inside_tag = False
            i += 1
            continue
        if not inside_tag and cur.isalpha() and len(cur) > 1:  # if reaches a valid word
            word = lemmatizer.lemmatize(cur.lower())  # lemmatize the lower-cased word
            if word not in stop_words:
                words.append(word)  # append to list of words if not a stop word
        i += 1
        continue
    return words


# Input: a part of an email
# Output: a list of meaningful words
def process_node(part):
    #check whether this part is multipart or not
    if part.is_multipart():
        words = list()
        # add children to bfs frontier
        for payload in part.get_payload():
            words.extend(process_node(payload))
    else:
        # parse the content and get a list of words
        content_type = part.get_content_type()
        if content_type == 'text/plain':
            words = clean_up_plain_text(part.get_payload())
        elif content_type == 'text/html':
            words = clean_up_html(part.get_payload())
        else:
            words = []
    return words

# Input: a string of path of an email
# Output: a list of words inside the email
def read_email(email_path):
    # open the email file for reading
    with open(email_path, 'r', encoding="utf8", errors='ignore') as f:
        raw = f.read()
        root = email.message_from_string(raw)
        # traverse and parse the multipart email until the end
        words = process_node(root)
        return words


# Input: a string of path of the label file
# Output: an array of labels, 1 is spam and 0 is ham
def read_labels(label_path):
    # open file for reading
    with open(label_path, 'r') as f:
        labels = f.readlines()
        # extract all labels
        labels = [label.partition(' ')[0] for label in labels]
        # convert to 0/1
        for i in range(len(labels)):
            if labels[i] == 'spam':
                labels[i] = 1
            else:
                labels[i] = 0
        labels = np.asarray(labels)
        return labels


# Input： model (naive bayes or svm)
#         size: size of training and validation data
def train_and_evaluate_model(model, size):
    email_path='trec07p/trec07p/data/inmail.'
    label_path='trec07p/trec07p/full/index'
    count_vect = CountVectorizer()
    emails = []
    for i in range(1, size+1):
        # read all the emails
        emails.append(' '.join(read_email(email_path+str(i))))
    # get raw count
    raw_counts = count_vect.fit_transform(emails)
    # get tf-idf
    transformer = TfidfTransformer().fit(raw_counts)
    tf_idf = transformer.transform(raw_counts)
    # get labels
    labels = read_labels(label_path)[:size]
    # apply a ten-fold cross validation on tf-idf
    kf = KFold(n_splits=10)
    kf.get_n_splits(tf_idf)
    # precision
    prec_total = 0.0
    # recall
    rec_total = 0.0
    # accuracy
    acc_total = 0.0
    # f1_measure
    f1_total = 0.0
    for train_index, test_index in kf.split(tf_idf):
        # get train split and test split
        X_train, X_test = tf_idf[train_index], tf_idf[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # train the model and make prediction
        model.fit(X_train,y_train)
        predicted = model.predict(X_test)
        # evaluate model by calculating precision, recall, accuracy and F-measure
        prec_total += precision_score(y_test, predicted)
        rec_total += recall_score(y_test, predicted)
        acc_total += accuracy_score(y_test, predicted)
        f1_total += f1_score(y_test, predicted)
    return prec_total/10, rec_total/10, acc_total/10, f1_total/10

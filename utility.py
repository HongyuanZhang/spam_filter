# to parse email
import email
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# machine learning tools
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

# for lemmatizing and getting rid of stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_up_plain_text(string):
    words=list()
    tokens_raw = nltk.word_tokenize(string)
    i = 0
    while i < len(tokens_raw):
        cur = tokens_raw[i]  # current token
        if cur.isalpha() and len(cur) > 1:
            word = lemmatizer.lemmatize(cur.lower())  # lemmatize the lower-cased word
            if word not in stop_words:
                words.append(word)
        i += 1
        continue
    return words

def clean_up_html(html):
    words=list()
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

def process_node(part):
    '''
    check whether this part is multipart or not
    '''
    if part.is_multipart():
        words=list()
        # add children to bfs frontier
        for payload in part.get_payload():
            words.extend(process_node(payload))
    else:
        # parse the content and get a list of words
        content_type=part.get_content_type()
        if content_type=='text/plain':
            words=clean_up_plain_text(part.get_payload())
        elif content_type=='text/html':
            words=clean_up_html(part.get_payload())
        else:
            words=[]
    return words

def read_email(email_path):
    '''
    takes an email path
    traverse the multipart email until the end like a tree using bfs
    returns an array of words inside the email 
    '''
    with open(email_path, 'r', encoding="utf8", errors='ignore') as f:
        raw=f.read()
        root=email.message_from_string(raw)
        words=process_node(root)
        return words

def read_labels(label_path):
    '''
    read all the email labels
    1 is spam and 0 is ham
    '''
    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [label.partition(' ')[0] for label in labels]
        for i in range(len(labels)):
            if labels[i] == 'spam':
                labels[i] = 1
            else:
                labels[i] = 0
        labels=np.asarray(labels)
        return labels

def train_and_evaluate_model(model,size):
    '''
    model is the machine learning model, naive bayes or svm
    size is the size of training and validation data
    we use a ten-fold cross validation
    '''
    email_path='trec07p/data/inmail.'
    label_path='trec07p/full/index'
    count_vect = CountVectorizer()
    emails = []
    for i in range(1,size+1):
        # read all the emails
        emails.append(' '.join(read_email(email_path+str(i))))
    # get raw count
    raw_counts = count_vect.fit_transform(emails)
    transformer = TfidfTransformer().fit(raw_counts)
    # get tfidf
    tf_idf = transformer.transform(raw_counts)
    labels = read_labels(label_path)[:size]
    classes=np.unique(labels)
    # get ten-fold splits
    kf = KFold(n_splits=10)
    kf.get_n_splits(tf_idf)
    acc_total=0.0
    rec_total=0.0
    for train_index, test_index in kf.split(tf_idf):
        # get train split and test split
        X_train, X_test = tf_idf[train_index], tf_idf[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # train the model and make prediction
        model.fit(X_train,y_train)
        predicted = model.predict(X_test)
        # get confusion matrix and calculate the accuracy and recall
        cm=confusion_matrix(y_test,predicted,labels=classes)
        tp=cm[0][0]
        fp=cm[1][0]
        fn=cm[0][1]
        acc=tp/(tp+fp)
        rec=tp/(tp+fn)
        acc_total+=acc
        rec_total+=rec
    #print the average accuracy and recall
    print("Accuracy:",acc_total/10)
    print("Recall:",rec_total/10)


if __name__ == '__main__':
    words=read_email('trec07p/data/inmail.1')
    print(words)

from utility import * 
from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix

def get_dict(emails):
    '''
    construct a dictionary out of all the emails in the set
    '''
    paths=['trec07p/data/inmail.'+str(i) for i in emails]
    words_list=[read_email(p) for p in paths]
    dictionary=sum(words_list,[])
    dictionary=list(set(dictionary))
    print(len(dictionary))
    return dictionary

def train_svm(dictionary,train_set):
    '''
    train a non-linear svm on training set (the index of the emails)
    return the trained svm
    '''
    paths=['trec07p/data/inmail.'+str(i) for i in train_set]
    words_list=[read_email(p) for p in paths]
    features_matrix = np.zeros((len(train_set),len(dictionary)))
    for i in range(len(words_list)):
        words=words_list[i]
        for w in words:
            j=dictionary.index(w)
            features_matrix[i][j]=words.count(w)
    labels=read_labels('trec07p/full/index')
    train_labels=[labels[a] for a in train_set]
    train_labels=np.asarray(train_labels)
    print(features_matrix)
    print(train_labels)

    #model=svm.NuSVC(gamma='auto')
    model=svm.LinearSVC()
    model.fit(features_matrix,train_labels)
    return model

def test_svm(model,dictionary,test_set):
    '''
    given the svm and a test_set (the index of the test emails)
    print the testing accuracy, recall and confusion matrix
    '''
    paths=['trec07p/data/inmail.'+str(i) for i in test_set]
    words_list=[read_email(p) for p in paths]
    features_matrix = np.zeros((len(test_set),len(dictionary)))
    for i in range(len(words_list)):
        words=words_list[i]
        for w in words:
            j=dictionary.index(w)
            features_matrix[i][j]=words.count(w)
    labels=read_labels('trec07p/full/index')
    test_labels=[labels[a] for a in test_set]
    test_labels=np.asarray(test_labels)

    result=model.predict(features_matrix)
    print(confusion_matrix(test_labels,result))

def classify_email(email_path):
    '''
    classifies an email to be spam or not
    return 1 for spam, 0 for non-spam
    '''

if __name__ == '__main__':
    '''
    result=classify_email('trec07p/data/inmail.1')
    print(result)
    '''
    dictionary=get_dict(range(1,10001))
    model=train_svm(dictionary,range(1,9001))
    test_svm(model,dictionary,range(9001,10001))

from utility import * 
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sizes = [500, 1000, 10000, 70000] # various data size
    nb_precs = [] # list for holding naive bayes model's precision
    nb_recs = [] # list for holding naive bayes model's recall
    nb_accs = [] # list for holding naive bayes model's accuracy
    nb_f1s = [] # list for holding naive bayes model's F1 score
    svm_precs = [] # list for holding svm model's precision
    svm_recs = [] # list for holding svm model's recall
    svm_accs = [] # list for holding svm model's accuracy
    svm_f1s = [] # list for holding svm model's F1 score

    for size in sizes:
        print("Size: ", size)
        nb_model = MultinomialNB()
        svm_model = svm.LinearSVC()
        # train and get evaluation measures
        nb_prec, nb_rec, nb_acc, nb_f1 = train_and_evaluate_model(nb_model, size)
        svm_prec, svm_rec, svm_acc, svm_f1 = train_and_evaluate_model(svm_model, size)
        # record evaluations
        nb_precs.append(nb_prec)
        nb_recs.append(nb_rec)
        nb_accs.append(nb_acc)
        nb_f1s.append(nb_f1)
        svm_precs.append(svm_prec)
        svm_recs.append(svm_rec)
        svm_accs.append(svm_acc)
        svm_f1s.append(svm_f1)
        print("Naive Bayes Model's Precision, Recall, Accuracy and F1-score: ")
        print(nb_prec, nb_rec, nb_acc, nb_f1)
        print("SVM Model's Precision, Recall, Accuracy and F1-score: ")
        print(svm_prec, svm_rec, svm_acc, svm_f1)
    # plot the two models' precision for comparison
    plt.subplot(2, 2, 1)
    plt.plot(sizes, nb_precs, '.-')
    plt.plot(sizes, svm_precs, '.-')
    plt.title('SVM and NB Model Performance Comparison')
    plt.xlabel('Data size (Train+Test)')
    plt.ylabel('Precision')
    plt.legend(['NB Precision', 'SVM Precision'], loc='upper left')
    # plot the two models' recall for comparison
    plt.subplot(2, 2, 2)
    plt.plot(sizes, nb_recs, '.-')
    plt.plot(sizes, svm_recs, '.-')
    plt.xlabel('Data size (Train+Test)')
    plt.ylabel('Recall')
    plt.legend(['NB Recall', 'SVM Recall'], loc='upper left')
    # plot the two models' accuracy for comparison
    plt.subplot(2, 2, 3)
    plt.plot(sizes, nb_accs, '.-')
    plt.plot(sizes, svm_accs, '.-')
    plt.xlabel('Data size (Train+Test)')
    plt.ylabel('Accuracy')
    plt.legend(['NB Accuracy', 'SVM Accuracy'], loc='upper left')
    # plot the two models' F1 score for comparison
    plt.subplot(2, 2, 4)
    plt.plot(sizes, nb_f1s, '.-')
    plt.plot(sizes, svm_f1s, '.-')
    plt.xlabel('Data size (Train+Test)')
    plt.ylabel('F1-score')
    plt.legend(['NB F1-score', 'SVM F1-score'], loc='upper left')

    plt.show()
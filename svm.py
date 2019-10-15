from utility import * 
from sklearn import svm

if __name__ == '__main__':
    # different training sizes
    for size in [500,1000,10000,70000]:
        print("Size:",size)
        model=svm.LinearSVC()
        train_and_evaluate_model(model,size)
        print("")

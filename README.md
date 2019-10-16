# spam filter
SVM/Naive Bayes Spam Classifier

# files
utility.py - contains methods for parsing emails and reading labels, also has a method for training and evaluating machine learning models

models.py - calls the training and evaluation method in the utility module to train and evaluate a Naive Bayes Model and a SVM model while generating a plot for visualizing evaluation results (precision, recall, accuracy and F1 score)

# usage
Call models.py after having downloaded and decompressed the TREC 2007 Spam Corpus in the same folder as the code files. If you are still given an error saying that Python cannot find target files for opening/reading, check relative paths of the email files and the label file, and change the email_path and label_path variables in the code to appropriate path strings.

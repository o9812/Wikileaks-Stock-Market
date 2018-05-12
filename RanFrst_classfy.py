import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
from scipy.sparse import hstack
import sys
import os
import glob
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RF_classfication:
    def __init__(self, file_path="./Final_merge/final_Single_australia"):
        # initial all data: train , validation and test set
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.size = self.loaddata(file_path)

        self.X_train_tfidf, self.X_valid_tfidf, self.X_test_tfidf = self.text_sparse(self.X_train, self.X_valid, self.X_test, ngram_range=(1, 2))
        self.x_arr2matrix_train = None
        self.x_arr2matrix_valid = None

    def loaddata(self, file_path):
        '''
        input: file path
        output:
            return train, validation, test set.
        '''
        # load data and drop na
        austrilia = pd.read_json(file_path)
        # change the str list to array

        def list2arry(x):
            line = re.sub("[!@#$\n'']", '', x).replace("[", "").replace("]", "").strip().split(" ")
            try:
                return np.asarray([float(i) for i in line if i != ''])
            except:
                return None

        def filter_y(y):
            try:
                return float(y)
            except:
                return None
        # drop the none
        austrilia['lg_rt_features'] = austrilia['lg_rt_features'].apply(lambda x: list2arry(x)).values
        austrilia['num_lable'] = austrilia['num_lable'].apply(lambda x: filter_y(x)).values
        austrilia = austrilia.dropna()
        size = austrilia.shape

        x_austrilia = austrilia[['date', 'Content', 'lg_rt_features']]
        y_austrilia = austrilia[['date', 'dummy_lable', 'num_lable']]

        # split data as train(0.8), validation(0.1) and test set(0.1)
        X_train, X_valid, y_train, y_valid = train_test_split(x_austrilia, y_austrilia, test_size=0.2, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=42)

        return X_train, y_train, X_valid, y_valid, X_test, y_test, size

    def text_sparse(self, X_train, X_valid, X_test, ngram_range=(1, 2)):
        '''
        input
            X_train, X_valid
        output:
            X_train_tfidf, X_valid_tfidf
        '''
        # fit the traing data for vector
        tfidf_vectorizer = TfidfVectorizer(binary=True, ngram_range=ngram_range)
        tfidf_vectorizer.fit(X_train["Content"])
        # turn text to matrix
        X_train_tfidf = tfidf_vectorizer.transform(X_train["Content"])
        X_valid_tfidf = tfidf_vectorizer.transform(X_valid["Content"])
        X_test_tfidf = tfidf_vectorizer.transform(X_test["Content"])

        return X_train_tfidf, X_valid_tfidf, X_test_tfidf

    def rf_text(self, X_train_tfidf, y_train, X_valid_tfidf, y_valid, n_estimators=10):
        clf_text = RandomForestClassifier(n_estimators)
        clf_text.fit(X_train_tfidf, y_train['dummy_lable'])

        # calculate the mse
        y_predict_text = clf_text.predict(X_valid_tfidf)

        fpr, tpr, _ = roc_curve(y_valid['dummy_lable'], y_predict_text)
        roc_auc = auc(fpr, tpr)

        return ['fpr is:' + str(fpr), 'tpr is:' + str(tpr), 'roc_auc is: ' + str(roc_auc), self.draw(fpr, tpr, roc_auc, 'text features')]

    def rf_price(self, X_train, y_train, X_valid, y_valid, n_estimators=10):
        self.x_arr2matrix_train = np.array(X_train['lg_rt_features'].tolist())
        clf_price = RandomForestClassifier(n_estimators)
        clf_price.fit(self.x_arr2matrix_train, y_train['dummy_lable'])

        # calculate the mse, let a column of arrays to matrix
        self.x_arr2matrix_valid = np.array(X_valid['lg_rt_features'].tolist())
        y_predict_price = clf_price.predict(self.x_arr2matrix_valid)
        fpr, tpr, _ = roc_curve(y_valid['dummy_lable'], y_predict_price)
        roc_auc = auc(fpr, tpr)

        return ['fpr is:' + str(fpr), 'tpr is:' + str(tpr), 'roc_auc is: ' + str(roc_auc), self.draw(fpr, tpr, roc_auc, 'numerical features')]

    def rf_mix(self, X_train, y_train, X_valid, y_valid, n_estimators=10):
        # mix sparse matrix(tf-idf) and numpy array (price)
        x_mix_train = hstack((self.x_arr2matrix_train, self.X_train_tfidf))
        clf_mix = RandomForestClassifier(n_estimators)
        clf_mix.fit(x_mix_train, y_train['dummy_lable'])

        # x_arr2matrix_valid = np.array(X_valid['lg_rt_features'].tolist())
        x_mix_valid = hstack((self.x_arr2matrix_valid, self.X_valid_tfidf))
        y_predict_mix = clf_mix.predict(x_mix_valid)
        fpr, tpr, _ = roc_curve(y_valid['dummy_lable'], y_predict_mix)
        roc_auc = auc(fpr, tpr)

        return ['fpr is:' + str(fpr), 'tpr is:' + str(tpr), 'roc_auc is: ' + str(roc_auc), self.draw(fpr, tpr, roc_auc, 'mix features')]

    def draw(self, fpr, tpr, roc_auc, title):
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic graph: ' + title)
        plt.legend(loc="lower right")
        # return plt object
        return plt


def write_file(path, fileName, data):
    # path to store figure
    figure_path = path + '/' + fileName + '_figure'
    if not os.path.exists(figure_path):
        '''
        if there is no output directory, creating one
        '''
        # try:
        #     os.makedirs(path)

        # create figures
        os.makedirs(figure_path)

    filePathNameWExt = './' + path + '/' + fileName
    with open(filePathNameWExt, 'w') as fp:
        # data.to_json(fp)
        fp.write("%s\n" % ('Country: ' + fileName))
        fp.write("\n")
        for i, thelist in enumerate(data[:-1]):
            for item in thelist[:-1]:
                fp.write("%s\n" % item)
            # add a empyt line
            fp.write("\n")
            # save the figure
            thelist[-1].savefig(figure_path + '/' + str(i))
            thelist[-1].close()
        fp.write("%s\n" % data[-1])
    fp.close()

    figure_path


if __name__ == "__main__":
    """
    run this program by:
        python RanFrst_classfy.py 100 ./data_country/ country_class_30 -country
        python RanFrst_classfy.py 100 ./data_year/ year_class_30 -year
    """
    n_estimators = int(sys.argv[1])
    # n_estimators = 1
    # relative path: ./output_yr/
    data_path = sys.argv[2]
    # output folder name
    output_filename = sys.argv[3]
    # type_rf is '-year' or '-country'
    type_rf = sys.argv[4]
    type_rf_ = None
    if type_rf == '-year':
        type_rf_ = 'yr*'
    elif type_rf == '-country':
        type_rf_ = 'final*'
    # output_filename = 'coutry'

    for file_path in glob.glob(data_path + type_rf_):
        print(file_path)
        fileName = file_path.split('_')[-1]
        print('%s: ' % type_rf, fileName)
        # ngram_range = sys.argv[3]
        rf = RF_classfication(file_path)

        # do text, price and mix random forest: return list of values
        only_text = rf.rf_text(rf.X_train_tfidf, rf.y_train, rf.X_valid_tfidf, rf.y_valid, n_estimators)
        only_price = rf.rf_price(rf.X_train, rf.y_train, rf.X_valid, rf.y_valid, n_estimators)
        mix_price_text = rf.rf_mix(rf.X_train_tfidf, rf.y_train, rf.X_valid_tfidf, rf.y_valid, n_estimators)
        # fileName

        write_file('output_' + output_filename, fileName, [only_text, only_price, mix_price_text, str(rf.size)])

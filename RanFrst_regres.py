import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
from scipy.sparse import hstack
import sys
import os
import glob
import pickle
from sklearn.feature_extraction import DictVectorizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RF_regression:
    def __init__(self, n_estimators, file_path="./Final_merge/final_Single_australia", ngram_range=(1, 2)):
        """
        initial the random forest class
        @ input par:
            n_estimators: the number of estimators
            file_path   : the path for final data.
                ex: ./Final_merge/final_Single_australia
            ngram_range : the number of estimators

        @ output:
            dataset: initial all data, train , validation and test set
        """
        self.n_estimators = n_estimators
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.size = self.loaddata(file_path)

        self.X_train_tfidf, self.X_valid_tfidf, self.X_test_tfidf = self.text_sparse(self.X_train, self.X_valid, self.X_test, ngram_range)

        self.x_arr2matrix_train = np.array(self.X_train['lg_rt_features'].tolist())
        self.x_arr2matrix_valid = np.array(self.X_valid['lg_rt_features'].tolist())
        self.x_mix_train = hstack((self.x_arr2matrix_train, self.X_train_tfidf))
        self.x_mix_valid = hstack((self.x_arr2matrix_valid, self.X_valid_tfidf))

    def loaddata(self, file_path):
        """
        data preprocessing for SVM model
        @ input:
            file path.
        @ output:
            return train, validation, test set.
        """
        # load data and drop na
        austrilia = pd.read_json(file_path)

        # change the str list to array
        def list2arry(x):
            """
            @ input:
                dataframe with special symbol.
            @ output:
                a clean dataframe, with type of np array.
            """
            line = re.sub("[!@#$\n'']", '', x).replace("[", "").replace("]", "").strip().split(" ")
            try:
                return np.asarray([float(i) for i in line if i != ''])
            except:
                return None

        def filter_y(y):
            """
            @ input:
                dataframe.
            @ output:
                if possible, let price data as float type.
                if not, mark the exception as None and return.
            """
            try:
                return float(y)
            except:
                return None

        # clean the data and drop na
        austrilia['lg_rt_features'] = austrilia['lg_rt_features'].apply(lambda x: list2arry(x)).values
        austrilia['num_lable'] = austrilia['num_lable'].apply(lambda x: filter_y(x)).values
        austrilia = austrilia.dropna()
        # store the size of data
        size = austrilia.shape
        # create x features
        x_austrilia = austrilia[['date', 'Content', 'lg_rt_features']]
        # create y lable
        y_austrilia = austrilia[['date', 'dummy_lable', 'num_lable']]

        # split data as train(0.8), validation(0.1) and test set(0.1)
        X_train, X_valid, y_train, y_valid = train_test_split(x_austrilia, y_austrilia, test_size=0.2, random_state=44)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=44)

        return X_train, y_train, X_valid, y_valid, X_test, y_test, size
    # try new features
    # def plot_coefficients(X, random_forest):
    #     feature_names = X.columns.values
    #     importance = random_forest.feature_importances

    #     sorted_importance = np.argsort(importance)

    #     # index of importance (top...)
    #     # least importance ~ most importance

    #     top10_importance = sorted_importance[-10:]

    #     top_names = feature_names[top10_importance]

    #     plt.xticks(top_names)
    #     plt.bar(top10_importance)

    def text_sparse(self, X_train, X_valid, X_test, ngram_range=(1, 2)):
        '''
        @ input
            dataframe: X_train, X_valid or X_test
            ngram_range: int of setting n-gram of text features
        @ output:
            X_train_tfidf: with vectorization TF-IDF, sparse matrix
        '''
        # fit the traing data for vector
        tfidf_vectorizer = TfidfVectorizer(binary=True, ngram_range=ngram_range)
        tfidf_vectorizer.fit(X_train["Content"])

        # with open(fileName + '.pickle', 'wb') as handle:
        #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # turn text to matrix
        X_train_tfidf = tfidf_vectorizer.transform(X_train["Content"])
        X_valid_tfidf = tfidf_vectorizer.transform(X_valid["Content"])
        X_test_tfidf = tfidf_vectorizer.transform(X_test["Content"])

        return X_train_tfidf, X_valid_tfidf, X_test_tfidf

    def rf_text(self, X_train_tfidf, y_train, X_valid_tfidf, y_valid, n_estimators=10):
        """
        only works for text feature
        @ input
            X_train_tfidf : sparse matrix from train data
            X_valid_tfidf: sparse matrix from valild data
            y_train: label of train data, include dummy and numerical lable
            y_valid: label of valid data, include dummy and numerical lable
            n_estimators: number of trees
        @ output:
            see the sklearn metric page
            http://scikit-learn.org/stable/modules/classes.html
            mse: string
            mae: string
            median_absolute_error: string
            r2_text: string
        """
        # Fit random forest
        clf_text = RandomForestRegressor(n_estimators)
        clf_text.fit(X_train_tfidf, y_train['num_lable'])

        # predict the value
        y_predict_text = clf_text.predict(X_valid_tfidf)
        # calculate the metrics
        mse_text = mean_squared_error(y_valid['num_lable'], y_predict_text)
        mae_text = mean_absolute_error(y_valid['num_lable'], y_predict_text)
        mdn_ae_text = median_absolute_error(y_valid['num_lable'], y_predict_text)
        r2_text = r2_score(y_valid['num_lable'], y_predict_text)

        print("mse_text is: ", mse_text)
        print("mae_text is: ", mae_text)
        print("median_absolute_error: ", mdn_ae_text)
        print("r2_text: ", r2_text)
        return ["mse_text is: " + str(round(mse_text, 8)), "mae_text is: " + str(round(mse_text, 8)), "median_absolute_error: " + str(round(mdn_ae_text, 8)), "r2_text: " + str(round(r2_text, 8))], clf_text

    def rf_price(self, X_train, y_train, X_valid, y_valid, n_estimators=10):
        """
        only works for price feature
        @ input
            X_train: train data
            X_valid: valild data
            y_train: label of train data, include dummy and numerical lable
            y_valid: label of valid data, include dummy and numerical lable
            n_estimators: number of trees
        @ output:
            see the sklearn metric page
            http://scikit-learn.org/stable/modules/classes.html
            mse: string
            mae: string
            median_absolute_error: string
            r2_text: string
        """

        # turn the data into matrix and feed to random forest for training data
        clf_price = RandomForestRegressor(n_estimators)
        clf_price.fit(self.x_arr2matrix_train, y_train['num_lable'])

        # predict vy the price feature
        y_predict_price = clf_price.predict(self.x_arr2matrix_valid)

        # calculate the metrics
        mse_price = mean_squared_error(y_valid['num_lable'], y_predict_price)
        mae_price = mean_absolute_error(y_valid['num_lable'], y_predict_price)
        mdn_ae_price = median_absolute_error(y_valid['num_lable'], y_predict_price)
        r2_price = r2_score(y_valid['num_lable'], y_predict_price)
        print("mse_price is: ", mse_price)
        print("mae_price is: ", mae_price)
        print("mdn_ae_price: ", mdn_ae_price)
        print("r2_price: ", r2_price)
        return ["mse_price is: " + str(round(mse_price, 8)), "mae_price is: " + str(round(mae_price, 8)), "mdn_ae_price: " + str(round(mdn_ae_price, 8)), "r2_price: " + str(round(r2_price, 8))], clf_price

    def rf_mix(self, X_train, y_train, X_valid, y_valid, n_estimators=10):
        """
        only works for mix feature;
        hstack two features, price and text
        @ input
            X_train: train data
            X_valid: valild data
            y_train: label of train data, include dummy and numerical lable
            y_valid: label of valid data, include dummy and numerical lable
            n_estimators: number of trees
        @ output:
            see the sklearn metric page
            http://scikit-learn.org/stable/modules/classes.html
            mse: string
            mae: string
            median_absolute_error: string
            r2_text: string
        """
        # mix sparse numpy array (price) and matrix(tf-idf) for train data

        clf_mix = RandomForestRegressor(n_estimators)
        clf_mix.fit(self.x_mix_train, y_train['num_lable'])

        # mix sparse numpy array (price) and matrix(tf-idf) for valid data
        y_predict_mix = clf_mix.predict(self.x_mix_valid)

        # calculate the metrics
        mse_mix = mean_squared_error(y_valid['num_lable'], y_predict_mix)
        mae_mix = mean_absolute_error(y_valid['num_lable'], y_predict_mix)
        mdn_ae_mix = median_absolute_error(y_valid['num_lable'], y_predict_mix)
        r2_mix = r2_score(y_valid['num_lable'], y_predict_mix)
        print("mse_mix is: ", mse_mix)
        print("mae_mix is: ", mae_mix)
        print("mdn_ae_mix: ", mdn_ae_mix)
        print("r2_mix: ", r2_mix)
        return ["mse_mix is: " + str(round(mse_mix, 8)), "mae_mix is: " + str(round(mae_mix, 8)), "mdn_ae_mix: " + str(round(mdn_ae_mix, 8)), "r2_mix: " + str(round(r2_mix, 8))], clf_mix


def write_file(path, fileName, data, models, plt_feature_important = None):
    """
    @ input:
        path        : open the file path to store file
        fileName    : the name of file should be called
        data        : store into file, here are four metircs
        models      : store models, here are three models. text, price and mix
    """
    if not os.path.exists(path + '/model'):
        '''
        if there is no output directory, creating one
        '''
        os.makedirs(path + '/model')

    filePathNameWExt = './' + path + '/' + fileName
    with open(filePathNameWExt, 'w') as fp:
        # iterate the file
        fp.write("%s\n" % ('Country: ' + fileName))
        fp.write("\n")
        for thelist in data[:-1]:
            for item in thelist:
                fp.write("%s\n" % item)
            fp.write("\n")
        fp.write("%s\n" % data[-1])
    fp.close()
    #  store the model
    if not os.path.exists(path):
        '''
        if there is no output directory, creating one
        '''
        os.makedirs(path)
    for i, model in enumerate(models):
        pickle.dump(model, open('./' + path + '/model/' + fileName + str(i) + ".p", "wb"))
    """
    comment this part for draw pictures
    """
    # plt_feature_important.savefig('./' + path + '/model/' + fileName + "_feature importance")
    # plt_feature_important.close()


def draw(importances, indices, top_50_list):
    """
    Function to draw plot of feature importance
    @input:
        importances : the features number
        indices     : the indices of features
        top_50_list : top 50 features
    @output:
        plt: plt objective of feature importance
    """
    plt.figure(figsize=(20, 16))
    plt.title("Feature importances")
    plt.bar(range(0, 50), importances[indices],
            color="r", align="center", width=0.5)
    plt.xticks(range(0, 50), top_50_list)
    plt.tick_params(axis='both', which='minor', labelsize=1)
    return plt


if __name__ == "__main__":
    """
    @input:
        argv[1]: the number of estimators
        argv[2]: input data path
        argv[3]: the ouput file name
        argv[4]: tree type
    @output:
        a folder: creating a "output folder with path, output_argv[3]"
        including for metrics and trained data

    run this program by:
        ex: python RanFrst_regres.py 30 ./data_all_country_neg/ country_1_neg -mix

    """
    # n_estimators = 1
    n_estimators = int(sys.argv[1])

    # example, a relative path: ./output_yr/
    data_path = sys.argv[2]

    # output folder name
    output_filename = sys.argv[3]
    # type_rf is '-price' or '-mix' or '-text' or '-all'
    tree_type = sys.argv[4]
    type_rf_ = '*'

    print('RanFrst Regression with ', n_estimators, ' estimators')
    for file_path in glob.glob(data_path + type_rf_):
        print(file_path)
        # deal with exception file name, negative suff
        fileName = file_path.split('_')[-1]
        if fileName == 'ng':
            fileName = file_path.split('_')[-2]
        print('%s: ' % fileName)

        rf = RF_regression(n_estimators, file_path)
        # list to store result
        result = []
        model_result = []
        # do text, price and mix random forest: return list of values
        # if the tag is -all or -text, train randomforest for text model
        if tree_type == '-text' or tree_type == '-all':
            only_text, model_text = rf.rf_text(rf.X_train_tfidf, rf.y_train, rf.X_valid_tfidf, rf.y_valid, n_estimators)
            # commet the following not to store intermediate result
            # store the model
            # result.append(only_text)
            # model_result.append(model_text)

            feature_importance = model_text.feature_importances_
            # Find best 50 features
            indices = np.argsort(feature_importance)[::-1][0:50]

            # temp store the training and validation set
            temp_x_train, temp_x_valid = rf.X_train_tfidf, rf.X_valid_tfidf

            # Select sparse Matrix, the top 50 features, renew the matrix
            rf.x_mix_train = rf.x_mix_train.tocsc()[:, indices]
            rf.x_mix_valid = rf.x_mix_valid.tocsc()[:, indices]

            # build models with selected matrix
            print('New Model')
            text_new, model_text_new = rf.rf_text(X_train, y_train, X_valid, y_valid)
            result.append(text_new)
            model_result.append(model_text_new)

            # restore the matrix from train, valid
            rf.X_train_tfidf, rf.X_valid_tfidf = temp_x_train, temp_x_valid

        # if the tag is -all or -price, train randomforest for price model
        if tree_type == '-price' or tree_type == '-all':
            only_price, model_price = rf.rf_price(rf.X_train, rf.y_train, rf.X_valid, rf.y_valid, n_estimators)
            # commet the following not to store intermediate result
            # store the model
            # result.append(only_price)
            # model_result.append(model_price)
            feature_importance = model_price.feature_importances_
            # Find best 50 features
            indices = np.argsort(feature_importance)[::-1][0:5]

            # temp store the training and validation set
            temp_x_train, temp_x_valid = rf.x_arr2matrix_train, rf.x_arr2matrix_valid
            # Select sparse Matrix, the top 50 features, renew the matrix
            rf.x_mix_train = rf.x_mix_train.tocsc()[:, indices]
            rf.x_mix_valid = rf.x_mix_valid.tocsc()[:, indices]

            # build models with selected matrix
            print('New Model')
            price_new, model_price_new = rf.rf_price(X_train, y_train, X_valid, y_valid)
            result.append(price_new)
            model_result.append(model_price_new)

            # restore the matrix from train, valid
            rf.x_arr2matrix_train, rf.x_arr2matrix_valid = temp_x_train, temp_x_valid

        # if the tag is -all or -mix, train randomforest for mix model
        if tree_type == '-mix' or tree_type == '-all':
            mix_price_text, model_mix = rf.rf_mix(rf.X_train_tfidf, rf.y_train, rf.X_valid_tfidf, rf.y_valid, n_estimators)

            # store the model
            # result.append(mix_price_text)
            # model_result.append(model_mix)

            # get feature imporatnce
            feature_importance = model_mix.feature_importances_
            # Find best 50 features
            indices = np.argsort(feature_importance)[::-1][0:50]

            # temp store the training and validation set
            temp_x_train, temp_x_valid = rf.x_mix_train, rf.x_mix_valid

            # Select sparse Matrix, the top 50 features, renew the matrix
            rf.x_mix_train = rf.x_mix_train.tocsc()[:, indices]
            rf.x_mix_valid = rf.x_mix_valid.tocsc()[:, indices]

            # build models with selected matrix
            print('New Model')
            mix_price_text_new, model_mix_new = rf.rf_mix(rf.X_train, rf.y_train, rf.X_valid, rf.y_valid)
            result.append(mix_price_text_new)
            model_result.append(model_mix_new)
            # restore the matrix from train, valid
            rf.x_mix_train, rf.x_mix_valid = temp_x_train, temp_x_valid

        # record the size
        print('size is: ' + str(rf.size))
        result.append(str(rf.size))
        write_file('output_' + output_filename, fileName, result, model_result)

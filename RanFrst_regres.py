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


class RF_regression:
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
        # x_austrilia['lg_rt_features'] = x_austrilia['lg_rt_features'].apply(lambda x: list2arry(x)).values
        y_austrilia = austrilia[['date', 'dummy_lable', 'num_lable']]

        # split data as train(0.8), validation(0.1) and test set(0.1)
        X_train, X_valid, y_train, y_valid = train_test_split(x_austrilia, y_austrilia, test_size=0.2, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=42)

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
        clf_text = RandomForestRegressor(n_estimators)
        clf_text.fit(X_train_tfidf, y_train['num_lable'])

        # calculate the mse
        y_predict_text = clf_text.predict(X_valid_tfidf)
        mse_text = mean_squared_error(y_valid['num_lable'], y_predict_text)
        mae_text = mean_absolute_error(y_valid['num_lable'], y_predict_text)
        mdn_ae_text = median_absolute_error(y_valid['num_lable'], y_predict_text)
        r2_text = r2_score(y_valid['num_lable'], y_predict_text)
        print("mse_text is: ", mse_text)
        print("mae_text is: ", mae_text)
        print("median_absolute_error: ", mdn_ae_text)
        print("r2_text: ", r2_text)
        return ["mse_text is: " + str(round(mse_text, 8)), "mae_text is: " + str(round(mse_text, 8)), "median_absolute_error: " + str(round(mdn_ae_text, 8)), "r2_text: " + str(round(r2_text, 8))]

    def rf_price(self, X_train, y_train, X_valid, y_valid, n_estimators=10):
        self.x_arr2matrix_train = np.array(X_train['lg_rt_features'].tolist())
        clf_price = RandomForestRegressor(n_estimators=10)
        clf_price.fit(self.x_arr2matrix_train, y_train['num_lable'])

        # calculate the mse, let a column of arrays to matrix
        self.x_arr2matrix_valid = np.array(X_valid['lg_rt_features'].tolist())
        y_predict_price = clf_price.predict(self.x_arr2matrix_valid)

        mse_price = mean_squared_error(y_valid['num_lable'], y_predict_price)
        mae_price = mean_absolute_error(y_valid['num_lable'], y_predict_price)
        mdn_ae_price = median_absolute_error(y_valid['num_lable'], y_predict_price)
        r2_price = r2_score(y_valid['num_lable'], y_predict_price)
        print("mse_price is: ", mse_price)
        print("mae_price is: ", mae_price)
        print("mdn_ae_price: ", mdn_ae_price)
        print("r2_price: ", r2_price)
        return ["mse_price is: " + str(round(mse_price, 8)), "mae_price is: " + str(round(mae_price, 8)), "mdn_ae_price: " + str(round(mdn_ae_price, 8)), "r2_price: " + str(round(r2_price, 8))]

    def rf_mix(self, X_train, y_train, X_valid, y_valid, n_estimators=10):
        # mix sparse matrix(tf-idf) and numpy array (price)
        x_mix_train = hstack((self.x_arr2matrix_train, self.X_train_tfidf))
        clf_mix = RandomForestRegressor(n_estimators)
        clf_mix.fit(x_mix_train, y_train['num_lable'])

        # x_arr2matrix_valid = np.array(X_valid['lg_rt_features'].tolist())
        x_mix_valid = hstack((self.x_arr2matrix_valid, self.X_valid_tfidf))
        y_predict_mix = clf_mix.predict(x_mix_valid)

        mse_mix = mean_squared_error(y_valid['num_lable'], y_predict_mix)
        mae_mix = mean_absolute_error(y_valid['num_lable'], y_predict_mix)
        mdn_ae_mix = median_absolute_error(y_valid['num_lable'], y_predict_mix)
        r2_mix = r2_score(y_valid['num_lable'], y_predict_mix)
        print("mse_mix is: ", mse_mix)
        print("mae_mix is: ", mae_mix)
        print("mdn_ae_mix: ", mdn_ae_mix)
        print("r2_mix: ", r2_mix)
        return ["mse_mix is: " + str(round(mse_mix, 8)), "mae_mix is: " + str(round(mae_mix, 8)), "mdn_ae_mix: " + str(round(mdn_ae_mix, 8)), "r2_mix: " + str(round(r2_mix, 8))]


def write_file(path, fileName, data):
    if not os.path.exists(path):
        '''
        if there is no output directory, creating one
        '''
        os.makedirs(path)

    filePathNameWExt = './' + path + '/' + fileName
    with open(filePathNameWExt, 'w') as fp:
        # data.to_json(fp)
        fp.write("%s\n" % ('Country: ' + fileName))
        fp.write("\n")
        for thelist in data[:-1]:
            for item in thelist:
                fp.write("%s\n" % item)
            fp.write("\n")
        fp.write("%s\n" % data[-1])
    fp.close()


if __name__ == "__main__":
    """
    run this program by:
        python RanFrst_regres_final.py 30 ./data_country/ country_30 -country
        python RanFrst_regres_final.py 30 ./data_year/ year_30 -year
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
        type_rf_ = '*'
    elif type_rf == '-country':
        type_rf_ = '*'
    # output_filename = 'coutry'
    print('RanFrst Regression with ' , n_estimators , ' estimators')
    for file_path in glob.glob(data_path + type_rf_):
        print(file_path)
        fileName = file_path.split('_')[-1]
        print('%s: ' % type_rf, fileName)
        # ngram_range = sys.argv[3]
        rf = RF_regression(file_path)

        # do text, price and mix random forest: return list of values
        only_text = rf.rf_text(rf.X_train_tfidf, rf.y_train, rf.X_valid_tfidf, rf.y_valid, n_estimators)
        only_price = rf.rf_price(rf.X_train, rf.y_train, rf.X_valid, rf.y_valid, n_estimators)
        mix_price_text = rf.rf_mix(rf.X_train_tfidf, rf.y_train, rf.X_valid_tfidf, rf.y_valid, n_estimators)
        # fileName

        write_file('output_' + output_filename, fileName, [only_text, only_price, mix_price_text, str(rf.size)])

import os
import sys
import numpy as np
import glob
import json
import pandas as pd
import csv
from datetime import datetime

# write file under the directory


def writeToJSONFile(path, fileName, data):
    '''
    merge different dictionaries from different files
    input:
        path: path to store the output
        fileName: the stored file name
        data: list of dictionaries
    output:
        21 countries output.
    '''

    if not os.path.exists(path):
        '''
        if there is no output directory, creating one
        '''
        os.makedirs(path)
    filePathNameWExt = './' + path + '/' + fileName
    with open(filePathNameWExt, 'w') as fp:
        data.to_json(fp)

# DEAL WITH TIME


def convert_time(o_time):
    o_time = o_time.strip()
    date, clock = o_time.split(',', 1)
    datetime_object = datetime.strptime(date, '%Y %B %d')
    return int(datetime_object.strftime('%Y%m%d'))

def convert_time_ex(date):
    datetime_object = datetime.strptime(date, '%m/%d/%Y')
    return int(datetime_object.strftime('%Y%m%d'))

if __name__ == '__main__':
    '''
    input:
        exchange rate
        merged cable
    output:
        a merged big table
    '''
    # read exchange rate and text
    # exchange = pd.read_json(open('./data/merge_joined_labled.json').read())
    exchange = pd.read_csv('./data/Ex_change_rate.csv')
    exchange['date'] = exchange['date'].apply(convert_time_ex)
    text_merge = pd.read_json(open('./data/merge_dataframe.json').read())
    # rename the column as date
    text_merge = text_merge.rename(index=str, columns={"Date": 'date'})
    # change the date format
    text_merge['date'] = text_merge['date'].apply(convert_time)
    #  merge the different dataframes
    read_merge = pd.merge(text_merge, exchange, on='date', how='inner')

    # country list
    coutry_list = ['australia', 'brazil', 'canada', 'china', 'denmark', 'hong kong', 'india', 'japan', 'korea', 'malaysia', 'mexico', 'new zealand', 'norway', 'sweden', 'south africa', 'singapore', 'sri lanka', 'switzerland', 'taiwan', 'thailand', 'united kingdom', 'venezuela']

    # trainset = pd.DataFrame(columns=['date', 'Content', 'num_lable', 'dummy_lable'])
    trainset = []
    for i in coutry_list:
        try:
            print('start: ', i)
            country = read_merge.loc[read_merge[i + '_x'] == 1]
            test_c = country[['date', 'Content', i + '_y', i + '_label']]
            test_c = test_c.rename(index=str, columns={i + '_y': 'num_lable', i + '_label': 'dummy_lable'})
            writeToJSONFile('./output', 'final_' + i, test_c)
            trainset.append(test_c)
            print('end: ', i)
        except:
            print('No such country:', i)

    result = pd.concat(trainset)
    writeToJSONFile('./output', 'final_All_countries', result)

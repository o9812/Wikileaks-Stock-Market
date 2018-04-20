import os
import sys
import numpy as np
import glob
import json
import pandas as pd
import csv

def writeToJSONFile(path, fileName, data):
    '''
    merge different dictionaries from different files
    input:
        path: path to store the output
        fileName: the stored file name
        data: list of dictionaries
    output:
        merged dictionaries(dataframe, if using pandas read)
    '''

    if not os.path.exists(path):
        '''
        if there is no output directory, creating one
        '''
        os.makedirs(path)
    filePathNameWExt = './' + path + '/' + fileName
    with open(filePathNameWExt, 'w') as fp:
        data.to_json(fp)


exchange = pd.read_json(open('./data/merge_joined_labled.json').read())
text_merge = pd.read_json(open('./output/merge_sql.json').read())
result = pd.merge(text_merge, exchange, on='date', how='outer')


coutry_list = ['australia', 'brazil', 'canada', 'china', 'denmark', 'hong kong', 'india', 'japan', 'korea', 'malaysia', 'mexico', 'new zealand', 'norway', 'sweden', 'south africa', 'singapore', 'sri lanka', 'switzerland', 'taiwan', 'thailand', 'united kingdom', 'venezuela']

trainset = pd.DataFrame(columns=['date', 'concanate_vec', 'label'])
for i in coutry_list:
    st = i
    try:
        australia_x = result[pd.notnull(result[st + '_x'])]
        concanate_vec = australia_x[st + '_vec'] + australia_x[st + '_logdff_vec'] + australia_x[st + '_x']
        label = australia_x[st + '_label']
        date = australia_x['date']
        tt = pd.DataFrame()
        tt['date'] = date
        tt['concanate_vec'] = concanate_vec
        tt['label'] = label
        trainset = pd.concat([trainset, tt])
    except:
        print('there is not:', i)

writeToJSONFile('./output', 'trainset.json', trainset)

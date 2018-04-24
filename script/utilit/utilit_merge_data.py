import os
import numpy as np
import glob
import json
import pandas as pd
'''
This is for merge two file
'''
'''
input:
    1. exchang price and vecotor
    2. me
output:
    3.
'''


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


read_merge = pd.read_json(open('./data/output/merge_dataframe.json').read())

sri = read_merge.loc[read_merge['sri lanka'] == 1]
test = sri[['Date', 'doc2vec']]
test['doc2vec'] = [np.asarray(i) for i in test['doc2vec']]
test = test[['Date', 'doc2vec']].groupby('Date').sum() / test.groupby(['Date']).count()
restult = test.reset_index()
restult = restult.rename(index=str, columns={"doc2vec": 'sri lanka'})

# coutry_list = ['australia', 'brazil', 'canada', 'china', 'denmark', 'hong kong', 'india', 'japan', 'korea', 'malaysia', 'mexico', 'new zealand', 'norway', 'sweden', 'south africa', 'singapore', 'sri lanka', 'switzerland', 'taiwan', 'thailand', 'united kingdom', 'venezuela']
coutry_list = ['australia', 'brazil', 'canada', 'china', 'denmark', 'hong kong', 'india', 'japan', 'korea', 'malaysia', 'mexico', 'new zealand', 'norway', 'sweden', 'south africa', 'singapore', 'switzerland', 'taiwan', 'thailand', 'united kingdom', 'venezuela']
for i in coutry_list:
    try:
        country = read_merge.loc[read_merge[i] == 1]
        test_c = country[['Date', 'doc2vec']]
        test_c['doc2vec'] = [np.asarray(i) for i in test_c['doc2vec']]
        test_c = test_c[['Date', 'doc2vec']].groupby('Date').sum() / test_c.groupby(['Date']).count()
        test_c = test_c.reset_index()
        test_c = test_c.rename(index=str, columns={"doc2vec": i})
        restult = pd.merge(restult, test_c, on='Date', how='outer')
    except:
        print('No such country:', i)
restult = restult.rename(index=str, columns={"Date": 'date'})
writeToJSONFile('./output', 'merge_sql.json', restult)

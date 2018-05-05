import os
import sys
import glob
import json
import pandas as pd


def writeToJSONFile(path, fileName, data):
    '''
    merge different file (dictionaries form) from different files
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


if __name__ == '__main__':
    print('start merge the file to dataframe')
    list_dict = []
    for file_name in glob.glob('data_match*'):
        config = json.loads(open(file_name).read())
        print('start %s' % (file_name))
        for key, value in config.items():
            list_dict.append(config[key])
        print('done %s' % (file_name))

    list2dataframe = pd.DataFrame.from_dict(list_dict)
    writeToJSONFile('./output_mergedasDF', 'merge_dataframe.json', list2dataframe)
    # using pandas to read json as dataframe
    # config = pd.read_json(open('./data/dataframe.json').read())

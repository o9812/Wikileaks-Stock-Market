import os
import sys
import csv
import itertools
from operator import itemgetter
from collections import defaultdict
import glob
import json


def split_dict_equally(input_dict, chunks=300):
    '''
    Splits dict by keys. Returns a list of dictionaries.
    '''
    # prep with empty dicts
    return_list = [dict() for idx in range(chunks)]
    idx = 0
    for k, v in input_dict.items():
        return_list[idx][k] = v
        if idx < chunks - 1:  # indexes start at 0
            idx += 1
        else:
            idx = 0
    return return_list


def writeToJSONFile(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    '''
    split the file into equal file
    '''
    print('start split')
    for file_name in glob.glob('*.json_tokenized.json'):
        config = json.loads(open(file_name).read())
        print('start %s' % (file_name))
        dict_list = split_dict_equally(config, 10)
        print('done %s' % (file_name))
        for inx, split_file in enumerate(dict_list):
            writeToJSONFile('./', file_name + str(inx), split_file)
        print('split done!')

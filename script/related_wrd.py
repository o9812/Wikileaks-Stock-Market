import os
import sys
import csv
import random
import itertools
from operator import itemgetter
from collections import defaultdict
import numpy as np
from numpy.linalg import svd
import utils
import glob
import json

'''
Modifed from Nishant Subramani's work
which is contributed to DSGA 1012 NLU
'''


def vector_length(u):
    '''
    input: one vector, numpy array
    output: the length of the array
    '''
    return (np.sum(u**2))**(1 / 2)


def cosine(u, v):
    '''
    input: two vecors, numpy array
    output: the length of the array
    '''
    return 1 - (np.dot(u, v)) / (vector_length(u) * vector_length(v))


def wrd2vector(word, mat, rownames):
    """Tool for finding the nearest neighbors of `word` in `mat` according
    to `distfunc`. The comparisons are between row vectors.

    Parameters
    ----------
    word : str
        The anchor word. Assumed to be in `rownames`.

    mat : np.array
        The vector-space model.

    rownames : list of str
        The rownames of mat.

    """
    try:
        w = mat[rownames.index(word)]
        return w
    except:
       # print('%s is not in the glove' % word)
        return False


def writeToJSONFile(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    '''
    :
    1. read list of company or country
    input:
        a tokenized document
    output:
        a bag of text from text
        or a summation of vector of a document
    '''
    #glv = utils.build_glove(os.path.join('glove.6B.50d.txt'))
    print('start glove')
    glv = utils.build_glove(os.path.join('glove.42B.300d.txt'))
    print('end glove')
    for file_name in glob.glob('*.json_tokenized.json*'):
        config = json.loads(open(file_name).read())
        print('start %s' % (file_name))
        for key, value in config.items():
            vector_lst = []
            doc_vec = 0
            count = 0
            for i in config[key]['Content']:
                vector = wrd2vector(word=i, mat=glv[0], rownames=glv[1])
                if vector is False:
                    continue
                doc_vec += vector
                count += 1
                vector_lst.append(vector.tolist())
            config[key]['text2vec'] = vector_lst
            try:
                config[key]['doc2vec'] = (doc_vec / count).tolist()
            except:
                config[key]['doc2vec'] = []
                print("No vec")
                #config[key]['doc2vec'] = []
        print('done %s' % (file_name))
        writeToJSONFile('./', file_name + '_text2vec', config)

        print('One work done!')

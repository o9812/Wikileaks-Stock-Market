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
    count = 0
    for file_name in glob.glob('data_crawled_2000*'):
        config = json.loads(open(file_name).read())
        print('start %s' % (file_name))
        for key, value in config.items():
            count += 1
        print('done %s ' % (file_name))
        print('count is ', count)

    print('all counts are', count)

import nltk
import json
# read file from directory
import glob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
nltk.download('punkt')

# clean and tokenize the document


def clean_tokenize(data):
    text = []
    text.append(data)
    try:
        text = "".join(text)
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        # remove all tokens that are not alphabetic
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
    except:
        words = []
    return words

# save as JSON file


def writeToJSONFile(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)

# DEAL WITH TIME format


def convert_time(date):
    o_time = config[key]['Date']
    o_time = o_time.strip()
    date, clock = o_time.split(',', 1)
    datetime_object = datetime.strptime(date, '%Y %B %d')
    return datetime_object.strftime('%Y%m%d')


# main function to run
for file_name in glob.glob('*.json'):
    config = json.loads(open(file_name).read())
    for key, value in config.items():
        try:
            config[key]['Content'] = clean_tokenize(config[key]['Content'])
            config[key]['Raw content'] = clean_tokenize(config[key]['Raw content'])
            config[key]['Date'] = convert_time(config[key]['Date'])
        except:
            config[key]['Content'] = []

    # save file:
    writeToJSONFile('./', file_name + '_tokenized', config)
    print('One work done!')

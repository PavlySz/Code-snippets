'''Some useful RegEx'''
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Stemmer
ps = PorterStemmer()

# Stop words
STOP_WORDS = stopwords.words('english')

# Regex for emails
EMAIL_RE = re.compile(r'^([a-zA-Z0-9_\-\.]+)\
                      @((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)\
                          |(([a-zA-Z0-9\-]+\.)+))\
                              ([a-zA-Z]{2,4}|[0-9]{1,3})\
                                  (\]?)$')


# Regex for web addresses
WEB_ADDR_RE = re.compile(r'(https?:\/\/(?:www\.|(?!www))\
                         [a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}\
                             |www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\
                                 \.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))\
                                 [a-zA-Z0-9]+\.[^\s]{2,}\
                                     |www\.[a-zA-Z0-9]+\.[^\s]{2,})')


# Regex for punctuations/symbols
SYMBOLS_RE = re.compile(r'[^\w\s]')

# Leading and trailing white spaces
LEAD_TRAIL_SPACE_RE = re.compile(r'^\s+|\s+$')

# Excessive white spaces
SPACE_RE = re.compile(r'\s+')

# Numbers and decimal numbers
DEC_RE = re.compile(r'[0-9]+(\.[0-9]+)?')


def clean_text(dfs):
    '''
    Clean text by applying the above RegExs

    Args:
        dfs: categorical dataframe series to clean

    Returns:
        text: clean dataframe series
    '''
    text = dfs

    # Lowercase
    text = text.apply(lambda x: x.lower())

    # Remove e-mail adresses
    text = text.apply(lambda x: EMAIL_RE.sub('', x))

    # Remove web addesses
    text = text.apply(lambda x: WEB_ADDR_RE.sub('', x))

    # Remove symbols
    text = text.apply(lambda x: SYMBOLS_RE.sub('', x))

    # Remove numbers
    text = text.apply(lambda x: DEC_RE.sub('', x))

    # Remove leading and trailing whitespaces
    text = text.apply(lambda x: LEAD_TRAIL_SPACE_RE.sub('', x))

    # Remove excessive whitespaces in the middle
    text = text.apply(lambda x: SPACE_RE.sub(' ', x))

    # Remove stop words
    # Token each sentence (row)
    # Create a list of the tokens if these tokens are not stop words
    # Convert each row (which is a list of tokens) back to a string (a sentence)
    text = text.apply(lambda x: ' '.join([t for t in word_tokenize(x) if t not in STOP_WORDS]))

    # Stemming
    text = text.apply(lambda x: ' '.join([ps.stem(t) for t in word_tokenize(x)]))

    # Return the clean text
    return text

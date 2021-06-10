import re
import emojis
import enchant
import pandas as pd
import stanza

from string                             import punctuation
from nltk.corpus                        import stopwords
from nltk                               import word_tokenize

# initialize function to read files used to 
# preprocess only once at the beggining
def initialize():
    stNLP = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='es', use_gpu=True, logging_level='FATAL') 
    abbreviations = read_abbreviations()
    emojis = read_emojis()
    emoticons = read_emoticons()
    stopwords = read_stopwords()
    d_es = load_dictionary()

    return stNLP, abbreviations, emojis, emoticons, stopwords, d_es


# convert words into lemmas
def lemmatize_stanza(text, stNLP):
    if text == "":
        doc = stNLP("el")
    else:
        doc = stNLP(text)
    
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]

    print(lemmas)
    
    return lemmas

# remove url from text
def remove_url(text):
    return re.sub(r'http\S+', ' ', text)

# remove user mentions from tweets
def remove_mention(text):
    return re.sub(r'@([A-Za-z0-9_]+)', ' ', text)

# remove numbers from text
def remove_numbers(text):
    return re.sub(r'\d+', ' ', text)

# replace laughs for 'jajaj' to generalize
def replace_laugh(text, d_es):
    words = []
    tokens = text.split()

    # checks if exists in the dictonary and only replace it if not
    for word in tokens:
        if d_es.check(word):
            words.append(word)

        else:
            if (word.startswith('ja') | word.startswith('je') | word.startswith('ji') | word.startswith('jo')):
                words.append('jajaja')

    return ' '.join(words)


# remove punctuation marks
def remove_punctuation(text):
    # Get punctuation symbols
    punctuation_es = list(punctuation)
    
    # Add spanish punctuation
    punctuation_es.extend(['¿', '¡'])
    punctuation_es.extend(map(str,range(10)))
    
    # Apply removing
    text = ''.join([c for c in text if c not in punctuation_es])
    
    return text

# load Spanish dictionary
def load_dictionary():
    d_es = enchant.Dict("es_ES")
    enchant.request_dict(tag="es")
    d_es.add("coronavirus")
    d_es.add("covid")
    d_es.add("negacionista")
    d_es.add("jajaja")
    return d_es

# remove repetead characters from words
def remove_repeated_characters(text, d_es):    
    words = []
    tokens = text.split()

    for word in tokens:
        if d_es.check(word):
            words.append(word)

        else:
            word = re.sub(r'(\w)\1+', r'\1', word)

            if d_es.check(word):
                words.append(word)
    
    return ' '.join(words)

# read abbreviations file
def read_abbreviations():
    # Get abbreviations
    abbreviations = pd.read_csv('data/preprocessing/abbreviations.csv', 
                        header = None, 
                        index_col = 0,
                        squeeze = True).to_dict()
    
    return abbreviations

# replace abbreviations for their complete form
def replace_abbreviations(text, abbreviations):    
    # Replace abbreviations
    for k, v in abbreviations.items():
        if k in text:
            text = text.replace(k,v)

    return text

# read emojis file
def read_emojis():
    # Get emoji conversion
    emoji_list = pd.read_csv('data/preprocessing/emojis.csv', 
                        header = 0, 
                        index_col = 'emoji',
                        usecols = ['emoji', 'label'],
                        encoding = 'utf-8',
                        squeeze = True).to_dict()

    return emoji_list

# replace emojis for text meaning the emotion that represents
def replace_emojis_label(text, emoji_list):    
    # Get text emojis
    text_emojis = []    
    text_emojis = emojis.get(text)

    # Replace emojis
    for em in text_emojis:
        if em in emoji_list:
            text = text.replace(em, f' {emoji_list.get(em)} ')
        else:
            text = text.replace(em, ' ')

    return text

# read emoticons file
def read_emoticons():
    # Get emoticons and its labels
    emoticons = pd.read_csv('data/preprocessing/emoticons.csv', 
                        header = 0, 
                        index_col = 'emoticon',
                        usecols = ['emoticon', 'label'],
                        encoding = 'utf-8',
                        squeeze = True).to_dict()
    
    return emoticons

# replace emoticons for text meaning the emotion that represents
def replace_emoticons_label(text, emoticons):
    # Replace emoticons by its label
    for em in emoticons:
        em = (f' {em} ')
        if em in text:
            text = text.replace(em, f' {emoticons.get(em)} ')

    return text

# remove extra blank spaces from text
def remove_extra_spaces(text):
    return re.sub(' +', ' ', text)

# read stopwords file
def read_stopwords():
    stopwords_es = stopwords.words('spanish')
    stopwords_es.remove('no')
    stopwords_es.remove('muy')
    stopwords_es.remove('mucho')
    stopwords_es.remove('poco')
    stopwords_es.append('ed')

    return stopwords_es

# remove stopwords from text
def remove_stopwords(text, stopwords_es):
    tokens_new = []
    tokens_old = word_tokenize(text)

    for word in tokens_old:
        if word not in stopwords_es:
            tokens_new.append(word)
    
    text = ' '.join(tokens_new)
    
    return text


if __name__ == '__main__':
    main()
    
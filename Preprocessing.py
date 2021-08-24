import re
import emojis
import enchant
import pandas as pd
import stanza
import es_dep_news_trf

from string             import punctuation
from nltk.corpus        import stopwords
from nltk               import word_tokenize


# initialize function to read files used to 
# preprocess only once at the beggining
def initialize():
    stNLP = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='es', use_gpu=True)
    abbreviations = read_abbreviations()
    emojis = read_emojis()
    emoticons = read_emoticons()
    stopwords = read_stopwords()
    d_es = load_dictionary()

    return stNLP, abbreviations, emojis, emoticons, stopwords, d_es


# convert words into lemmas
def lemmatize_stanza(text, stNLP):
    doc = stNLP(text) 
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]

    return lemmas


# convert words into lemmas
def lemmatize_spacy(text):
    nlp = es_dep_news_trf.load()
    doc = nlp(text)

    print(f'lemmas info:')
    for w in doc:
        print(f'\ttext: {w.text}\tlemma: {w.lemma_}\tpos: {w.pos_}')

    lemmas = [w.lemma_ for w in doc]

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


# check if words exist in the dictionary
def check_dictionary(text, d_es):
    words = []
    tokens = text.split()

    # checks if exists in the dictonary
    for word in tokens:
        if d_es.check(word):
            words.append(word)

        else:
            # remove repeated characters and check if now is in the dictionary
            word = remove_repeated_characters(word)
            if d_es.check(word):
                words.append(word)

            # replace a laugh for 'jajaj' to generalize, if necessary
            if replace_laugh(word):         
                words.append('jajaja')

    return ' '.join(words)


# check if a word is a laugh 
def replace_laugh(word):
    return (word.startswith('jaj') | word.startswith('jej') | word.startswith('jij') | word.startswith('joj'))
    

# remove punctuation marks
def remove_punctuation(text):
    # Get punctuation symbols
    punctuation_es = list(punctuation)
    
    # Add spanish punctuation
    punctuation_es.extend(['¿', '¡'])
    punctuation_es.extend(map(str,range(10)))
    
    # Apply removing
    for sign in punctuation_es:
        if sign in text:
            text = text.replace(sign, ' ')
    
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
def remove_repeated_characters(word):    
    return re.sub(r'(\w)\1+', r'\1', word)


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
        k = f' {k} '
        if k in text:
            text = text.replace(k, f' {v} ')
            print(f'key: {k}    value: {v}  text: {text}')

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
    for emoji in text_emojis:
        if emoji in emoji_list:
            text = text.replace(emoji, f' {emoji_list.get(emoji)} ')
        else:
            text = text.replace(emoji, ' ')

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
    for emoticon, emotion in emoticons.items():
        emoticon = f'{emoticon} '
        if emoticon in text:
            text = text.replace(emoticon, f' {emotion} ')

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
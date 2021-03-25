import re
import nltk
import emoji
import emojis
import regex
import spacy
import enchant, itertools
import pandas as pd
import stanza 
import xml.etree.ElementTree as ET

# stanza.download('es', package='ancora', processors='tokenize,mwt,pos,lemma', verbose=True) 
# nltk.download('stopwords')
# nltk.download('punkt')

from string                             import punctuation
from nltk.corpus                        import stopwords
from nltk.tokenize                      import sent_tokenize
from nltk.stem                          import SnowballStemmer, LancasterStemmer, PorterStemmer
from nltk                               import word_tokenize
from nltk.tokenize.treebank             import TreebankWordDetokenizer
from sklearn.feature_extraction.text    import strip_accents_unicode


def main():
    # test()
    # read_csv()
    read_tsv()

def test():
    stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es = initialize()
    text = "\"odio esto, es un puto infierno. covid vete YA!!! :( https://t.co/AHxQH2omqL\""
    text = text_preprocessing_debug(text, stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es)
    print(text)

def read_tsv():
    import csv
    import time

    stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es = initialize()
    
    with open('data/tweets/train.tsv', 'r', encoding='utf-8') as infile, open('data/tweets/train-processed.tsv', 'w', newline = '', encoding = 'utf-8') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
        line_count = 1
        print('START PROCESSING ...')
        writer.writerow(['id', 'event', 'tweet', 'offensive', 'emotion', 'processed_tweet'])


        for row in reader:
            new_row = row
            text, _ = text_preprocessing(row[2], stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es)
            new_row.append(text)
            writer.writerow(new_row)
            print('\tProcessing line ', line_count)
            line_count += 1

        outfile.flush() 

def read_csv():
    import csv
    import time
    start_time = time.time()
    stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es = initialize()

    with open('data/tweets/training_covid19.csv', 'r') as infile, open('data/tweets/training2.csv', 'w', newline = '', encoding = 'utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['id', 'tweet', 'processed_tweet', 'emotion', 'polarity'])

        line_count = 0
        print('START PROCESSING ...')

        for row in reader:
            new_row = [line_count, row[0], '', row[1], '']
            text, polarity = text_preprocessing(row[0], stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es)
            new_row[2] = text
            new_row[4] = polarity
            writer.writerow(new_row)
            
            print('\tProcessing line ', line_count)
            line_count += 1

        outfile.flush() 
        print(f'Execution time: {start_time - time.time()}')

def initialize():
    stNLP = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='es', use_gpu=True, logging_level='FATAL') 
    abbreviations = read_abbreviations()
    emojis = read_emojis()
    emoticons = read_emoticons()
    stopwords = read_stopwords()
    d_es = load_dictionary()
    senticon_es = load_senticon()

    return stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es

def text_preprocessing_debug(text, stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es):
    print("ORIGINAL TEXT: ", text, "\n")
    text = text.lower()
    print("LOWER TEXT: ", text, "\n")

    text = remove_url(text)
    print("REMOVE URL TEXT: ", text, "\n")
    text = remove_mention(text)
    print("REMOVE MENTION TEXT: ", text, "\n")
    text = remove_numbers(text)
    print("REMOVE NUMBERS TEXT: ", text, "\n")

    text = replace_emoticons_label(text, emoticons)
    print("REPLACE EMOTICONS TEXT: ", text, "\n")
    text = replace_emojis_label(text, emojis)
    print("REPLACE EMOJIS TEXT: ", text, "\n")
    text = replace_abbreviations(text, abbreviations)
    print("REPLACE ABBREV TEXT: ", text, "\n")

    text = replace_laugh(text)
    print("REPLACE LAUGH: ", text, "\n")
    text = remove_punctuation(text)
    print("REMOVE PUNCT TEXT: ", text, "\n")
    text = remove_repeated_characters(text, d_es)
    print("REMOVE REPT CHARS TEXT: ", text, "\n")

    # lemmas = lemmatize_spacy(text)
    lemmas = lemmatize_stanza(text, stNLP)
    text = TreebankWordDetokenizer().detokenize(lemmas)
    print("LEMMATIZE TEXT: ", text, "\n")

    text = remove_stopwords(text, stopwords)
    print("REMOVE STOPWORDS TEXT: ", text, "\n")

    polarity = text_polarity(text, senticon_es)

    text = remove_accents(text)
    print("REMOVE ACCENTS TEXT: ", text, "\n")
    
    text = remove_extra_spaces(text)   
    print("PROCESSED TEXT: ", text, "\n")

    return text, polarity

def text_preprocessing(text, stNLP, abbreviations, emojis, emoticons, stopwords, d_es, senticon_es):
    text = text.lower()

    text = remove_url(text)
    text = remove_mention(text)
    text = remove_numbers(text)

    text = replace_emoticons_label(text, emoticons)
    text = replace_emojis_label(text, emojis)
    text = replace_abbreviations(text, abbreviations)

    text = replace_laugh(text)
    text = remove_punctuation(text)
    text = remove_repeated_characters(text, d_es)

    # lemmas = lemmatize_spacy(text)
    lemmas = lemmatize_stanza(text, stNLP)
    text = TreebankWordDetokenizer().detokenize(lemmas)

    text = remove_stopwords(text, stopwords)
    polarity = text_polarity(text, senticon_es)
    text = remove_accents(text)

    text = remove_extra_spaces(text)   

    return text, polarity

def load_senticon():
    tree = ET.parse('data/preprocessing/senticon_es.xml')
    lemmas = tree.findall(".//lemma")
    
    senticon_es = {}
    for child in lemmas:
        senticon_es[child.text.strip()] = child.attrib.get('pol')
    
    return senticon_es

def text_polarity(text, senticon_es):
    polarity = 0.0
    tokens = text.split()

    for word in tokens:
        if word in senticon_es:
            polarity += float(senticon_es[word])

    return polarity

def lemmatize_spacy(text):
    nlp = spacy.load('es_core_news_lg')
    doc = nlp(text)
    lemmas = [tok.lemma_.lower() for tok in doc]
    return lemmas

def lemmatize_stanza(text, stNLP):
    if text == "":
        doc = stNLP("el")
    else:
        doc = stNLP(text)
    
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
    
    return lemmas

def remove_url(text):
    return re.sub(r'http\S+', ' ', text)

def remove_mention(text):
    return re.sub(r'@([A-Za-z0-9_]+)', ' ', text)

def remove_numbers(text):
    return re.sub(r'\d+', ' ', text)

def replace_laugh(text):
    text = re.sub(r'ja\S+', 'jajaja', text)
    text = re.sub(r'je\S+', 'jajaja', text)
    text = re.sub(r'jo\S+', 'jajaja', text)
    text = re.sub(r'ji\S+', 'jajaja', text)
    return text

def remove_punctuation(text):
    # Get punctuation symbols
    punctuation_es = list(punctuation)
    
    # Add spanish punctuation
    punctuation_es.extend(['¿', '¡'])
    punctuation_es.extend(map(str,range(10)))
    
    # Apply removing
    text = ''.join([c for c in text if c not in punctuation_es])
    
    return text

def load_dictionary():
    d_es = enchant.Dict("es_ES")
    enchant.request_dict(tag="es")
    d_es.add("coronavirus")
    d_es.add("covid")
    d_es.add("jajaja")
    return d_es

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

def remove_accents(text):
    return strip_accents_unicode(text)

def read_abbreviations():
    # Get abbreviations
    abbreviations = pd.read_csv('data/preprocessing/abbreviations.csv', 
                        header = None, 
                        index_col = 0,
                        squeeze = True).to_dict()
    
    return abbreviations

def replace_abbreviations(text, abbreviations):    
    # Replace abbreviations
    for k, v in abbreviations.items():
        if k in text:
            text = text.replace(k,v)

    return text

def replace_emojis(text):
    """ 
    # Get text emojis
    text_emojis = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            text_emojis.append(word)

    # Get emojis
    emojis = pd.read_csv('data/preprocessing/emojis.csv', 
                        header = 0, 
                        index_col = 'emoji',
                        usecols = ['emoji', 'name_spanish_formatted'],
                        encoding = 'utf-8',
                        squeeze = True).to_dict()
    
    print(emojis)

    # Replace emojis
    for em in text_emojis:
        text = text.replace(em, emojis.get(em)) 
    """
    text_emojis = []    
    text_emojis = emojis.get(text)

    # Get emojis
    emoji_list = pd.read_csv('data/preprocessing/emojis.csv', 
                        header = 0, 
                        index_col = 'emoji',
                        usecols = ['emoji', 'name_spanish_formatted'],
                        encoding = 'utf-8',
                        squeeze = True).to_dict()

    # Replace emojis
    for em in text_emojis:
        text = text.replace(em, emoji_list.get(em)) 

    return text

def read_emojis():
    # Get emoji conversion
    emoji_list = pd.read_csv('data/preprocessing/emojis-labeled-reduced.csv', 
                        header = 0, 
                        index_col = 'emoji',
                        usecols = ['emoji', 'label'],
                        encoding = 'utf-8',
                        squeeze = True).to_dict()

    return emoji_list

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

def read_emoticons():
    # Get emoticons and its labels
    emoticons = pd.read_csv('data/preprocessing/emoticons.csv', 
                        header = 0, 
                        index_col = 'emoticon',
                        usecols = ['emoticon', 'label'],
                        encoding = 'utf-8',
                        squeeze = True).to_dict()
    
    return emoticons

def replace_emoticons_label(text, emoticons):
    # Replace emoticons by its label
    for em in emoticons:
        em = (f' {em} ')
        if em in text:
            text = text.replace(em, f' {emoticons.get(em)} ')

    return text

def remove_extra_spaces(text):
    return re.sub(' +', ' ', text)

def read_stopwords():
    stopwords_es = stopwords.words('spanish')
    stopwords_es.remove('no')
    stopwords_es.remove('muy')
    stopwords_es.remove('mucho')
    stopwords_es.remove('poco')
    stopwords_es.append('ed')

    return stopwords_es

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
    
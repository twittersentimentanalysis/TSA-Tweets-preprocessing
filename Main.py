import re
import nltk
import emoji
import emojis
import regex
import spacy
import enchant, itertools
import pandas as pd
import stanza 

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
    # stNLP, abbreviations, emojis, emoticons, stopwords = initialize()
    # text = "�C�mo Ted Ed es fleufliw casaaa calla ayudar a las familias los estudiantes y profesores a navegar por el covid 19 ... pandemia a trav�s de - en Madurai �ndia"
    # text = text_preprocessing_debug(text, stNLP, abbreviations, emojis, emoticons, stopwords)
    # print(text)
    read_csv()

def read_csv():
    import csv
    import time
    start_time = time.time()
    stNLP, abbreviations, emojis, emoticons, stopwords = initialize()
    
    with open('data/tweets/covid-india-all.csv', 'r', encoding='utf-8') as infile, open('data/tweets/training-covid-india-all.csv', 'w', newline = '', encoding = 'utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        line_count = 1
        print('START PROCESSING ...')

        for row in reader:
            row[3] = text_preprocessing(row[2], stNLP, abbreviations, emojis, emoticons, stopwords)
            writer.writerow(row)
            
            print('\tProcessing line ', line_count)
            
            if line_count == 20000:
                print('DONE!')
                outfile.flush()                
                print("\n--- %s seconds ---" % (time.time() - start_time))
                break

            line_count += 1

def initialize():
    stNLP = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='es', use_gpu=True, logging_level='FATAL') 
    abbreviations = read_abbreviations()
    emojis = read_emojis()
    emoticons = read_emoticons()
    stopwords = read_stopwords()

    return stNLP, abbreviations, emojis, emoticons, stopwords

def text_preprocessing_debug(text, stNLP, abbreviations, emojis, emoticons, stopwords):
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

    text = remove_punctuation(text)
    print("REMOVE PUNCT TEXT: ", text, "\n")
    text = remove_repeated_characters(text)
    print("REMOVE REPT CHARS TEXT: ", text, "\n")

    # lemmas = lemmatize_spacy(text)
    lemmas = lemmatize_stanza(text, stNLP)
    text = TreebankWordDetokenizer().detokenize(lemmas)
    print("LEMMATIZE TEXT: ", text, "\n")

    text = remove_stopwords(text, stopwords)
    print("REMOVE STOPWORDS TEXT: ", text, "\n")
    text = remove_accents(text)
    print("REMOVE ACCENTS TEXT: ", text, "\n")
    
    text = remove_extra_spaces(text)   
    print("PROCESSED TEXT: ", text, "\n")

    return text

def text_preprocessing(text, stNLP, abbreviations, emojis, emoticons, stopwords):
    text = text.lower()

    text = remove_url(text)
    text = remove_mention(text)
    text = remove_numbers(text)

    text = replace_emoticons_label(text, emoticons)
    text = replace_emojis_label(text, emojis)
    text = replace_abbreviations(text, abbreviations)

    text = remove_punctuation(text)
    text = remove_repeated_characters(text)

    # lemmas = lemmatize_spacy(text)
    lemmas = lemmatize_stanza(text, stNLP)
    text = TreebankWordDetokenizer().detokenize(lemmas)

    text = remove_stopwords(text, stopwords)
    text = remove_accents(text)

    text = remove_extra_spaces(text)   

    return text

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

def remove_punctuation(text):
    # Get punctuation symbols
    punctuation_es = list(punctuation)
    
    # Add spanish punctuation
    punctuation_es.extend(['¿', '¡'])
    punctuation_es.extend(map(str,range(10)))
    
    # Apply removing
    text = ''.join([c for c in text if c not in punctuation_es])
    
    return text

def remove_repeated_characters(text):    
    d_es = enchant.Dict("es_ES")
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
            text = text.replace(em, emoji_list.get(em))
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
        if em in text:
            text = text.replace(em, ' ' + emoticons.get(em) + ' ')

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
    
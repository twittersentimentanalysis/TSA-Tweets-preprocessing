import nltk
import csv
import pandas as pd
import stanza
import Preprocessing

from nltk                               import word_tokenize
from nltk.tokenize.treebank             import TreebankWordDetokenizer

# download necessary packages
stanza.download('es', package='ancora', processors='tokenize,mwt,pos,lemma', verbose=True) 
nltk.download('stopwords')
nltk.download('punkt')

# main function
def main():
    # test()
    # read_csv()
    read_tsv()

# function for testing
def test():
    stNLP, abbreviations, emojis, emoticons, stopwords, d_es = Preprocessing.initialize()
    text = "@AntonioMautor Mi idea primera era hacerlo, pero como dijeron que posiblemente no había vacunas jejej para todo el mundo jijjj, tengo 45 jjaj y como jajjaj si el barco se hundiera, primero grupos de riesgo ... los mayores de 65, etc etc ... jersey"
    text = text_preprocessing_debug(text, stNLP, abbreviations, emojis, emoticons, stopwords, d_es)
    print(text)

# function to process tweets in a tsv format
def read_tsv():
    stNLP, abbreviations, emojis, emoticons, stopwords, d_es = Preprocessing.initialize()
    
    with open('data/tweets/emoevales_test.tsv', 'r', encoding='utf-8') as infile, open('data/tweets/test-processed_v3.tsv', 'w', newline = '', encoding = 'utf-8') as outfile:
        reader = csv.reader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
        writer = csv.writer(outfile, delimiter='\t')
        
        line_count = 1
        print('START PROCESSING ...')
        writer.writerow(['id', 'event', 'tweet', 'offensive', 'processed_tweet'])


        for row in reader:
            print('\tProcessing line ', line_count)
            new_row = row
            print(f'TEXT: {row[2]}')
            text = text_preprocessing(row[2], stNLP, abbreviations, emojis, emoticons, stopwords, d_es)
            new_row.append(text)
            writer.writerow(new_row)
            print(f'TEXT: {text}')
            line_count += 1

        outfile.flush() 

# function to process tweets in a csv format
def read_csv():
    stNLP, abbreviations, emojis, emoticons, stopwords, d_es = Preprocessing.initialize()

    with open('data/tweets/covid19-twitter-monitor-preprocessing.csv', 'r', encoding='utf-8') as infile, open('data/tweets/covid19-twitter-monitor-training-v2.csv', 'w', newline = '', encoding = 'utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['id', 'tweet', 'processed_tweet', 'emotion'])

        line_count = 0
        print('START PROCESSING ...')

        for row in reader:
            print(f'Row: {row}')
            new_row = [line_count, row[0], '', row[1]]
            print(f'New row: {new_row}')
            text = text_preprocessing(row[0], stNLP, abbreviations, emojis, emoticons, stopwords, d_es)
            new_row[2] = text
            writer.writerow(new_row)
            
            print('\tProcessing line ', line_count)
            line_count += 1

        outfile.flush() 


# function for testing
def text_preprocessing_debug(text, stNLP, abbreviations, emojis, emoticons, stopwords, d_es):
    print("ORIGINAL TEXT: ", text, "\n")
    text = text.lower()
    print("LOWER TEXT: ", text, "\n")

    text = Preprocessing.remove_url(text)
    print("REMOVE URL TEXT: ", text, "\n")
    text = Preprocessing.remove_mention(text)
    print("REMOVE MENTION TEXT: ", text, "\n")
    text = Preprocessing.remove_numbers(text)
    print("REMOVE NUMBERS TEXT: ", text, "\n")

    text = Preprocessing.replace_emoticons_label(text, emoticons)
    print("REPLACE EMOTICONS TEXT: ", text, "\n")
    text = Preprocessing.replace_emojis_label(text, emojis)
    print("REPLACE EMOJIS TEXT: ", text, "\n")
    text = Preprocessing.replace_abbreviations(text, abbreviations)
    print("REPLACE ABBREV TEXT: ", text, "\n")

    text = Preprocessing.replace_laugh(text, d_es)
    print("REPLACE LAUGH: ", text, "\n")
    text = Preprocessing.remove_punctuation(text)
    print("REMOVE PUNCT TEXT: ", text, "\n")
    text = Preprocessing.remove_repeated_characters(text, d_es)
    print("REMOVE REPT CHARS TEXT: ", text, "\n")

    lemmas = Preprocessing.lemmatize_stanza(text, stNLP)
    text = TreebankWordDetokenizer().detokenize(lemmas)
    print("LEMMATIZE TEXT: ", text, "\n")

    text = Preprocessing.remove_stopwords(text, stopwords)
    print("REMOVE STOPWORDS TEXT: ", text, "\n")
    
    text = Preprocessing.remove_extra_spaces(text)   
    print("PROCESSED TEXT: ", text, "\n")

    return text

# function for preprocessing
def text_preprocessing(text, stNLP, abbreviations, emojis, emoticons, stopwords, d_es):
    text = text.lower()

    text = Preprocessing.remove_url(text)
    text = Preprocessing.remove_mention(text)
    text = Preprocessing.remove_numbers(text)

    text = Preprocessing.replace_emoticons_label(text, emoticons)
    text = Preprocessing.replace_emojis_label(text, emojis)
    text = Preprocessing.replace_abbreviations(text, abbreviations)

    text = Preprocessing.replace_laugh(text, d_es)
    text = Preprocessing.remove_punctuation(text)
    text = Preprocessing.remove_repeated_characters(text, d_es)

    lemmas = Preprocessing.lemmatize_stanza(text, stNLP)
    text = TreebankWordDetokenizer().detokenize(lemmas)

    text = Preprocessing.remove_stopwords(text, stopwords)
    text = Preprocessing.remove_extra_spaces(text)   

    return text


if __name__ == '__main__':
    main()
    
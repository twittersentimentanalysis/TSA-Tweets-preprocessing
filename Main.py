import nltk
import csv
import pandas as pd
import stanza
import Preprocessing

from nltk.tokenize.treebank     import TreebankWordDetokenizer
from nltk.tokenize.treebank     import TreebankWordTokenizer

# download necessary packages
stanza.download('es', package='ancora', processors='tokenize,pos,lemma', verbose=True) 
nltk.download('stopwords')
nltk.download('punkt')

# main function
def main():
    test()
    # read_csv()
    # read_tsv()

# function for testing
def test():
    stNLP, abbreviations, emojis, emoticons, stopwords, d_es = Preprocessing.initialize()
    text = "\"Si ya he pasado el coronavirus, ¿para qué me vacuno? https://t.co/oZo4ZjSruk a través de @Conversation_E\""
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

    with open('data/tweets/covid19-india-dataset-translated.csv', 'r', encoding='utf-8-sig') as infile, open('data/tweets/covid19-india-dataset-training.csv', 'w', newline = '', encoding = 'utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['id', 'tweet', 'processed_tweet', 'emotion'])

        print('START PROCESSING ...')
        line_num = 0

        for row in reader:
            new_row = [line_num, row[2], '', row[3]]
            text = text_preprocessing(row[2], stNLP, abbreviations, emojis, emoticons, stopwords, d_es)
            new_row[2] = text
            writer.writerow(new_row)
            
            print(f'\tProcessing line {line_num}: {new_row}')
            line_num = line_num + 1

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

    text = Preprocessing.remove_punctuation(text)
    print("REMOVE PUNCT TEXT: ", text, "\n")
    text = Preprocessing.check_dictionary(text, d_es)
    print("DICTIONARY, REPLACE LAUGH, REMOVE REP CHARS: ", text, "\n")    

    # lemmas = Preprocessing.lemmatize_stanza(text, stNLP)
    lemmas = Preprocessing.lemmatize_spacy(text)
    print(lemmas)
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

    text = Preprocessing.remove_punctuation(text)
    text = Preprocessing.check_dictionary(text, d_es)  

    # lemmas = Preprocessing.lemmatize_stanza(text, stNLP)
    lemmas = Preprocessing.lemmatize_spacy(text)
    text = TreebankWordDetokenizer().detokenize(lemmas)

    text = Preprocessing.remove_stopwords(text, stopwords)
    text = Preprocessing.remove_extra_spaces(text)   

    return text


if __name__ == '__main__':
    main()
    
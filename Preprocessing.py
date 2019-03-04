# inspired by getenburg project
# https://github.com/jldbc/gutenberg/blob/master/preprocessing.py

import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import nltk
import string
import os
import regex as re
from collections import defaultdict



def punct_and_words(character_list, pos_file):
    """
    Iterate through all characters. Count periods, punctuation frequencies.
    word_count = words in sentence (resets to zero after a period).
    total_words is the book's total word count.
    """
    punct = ['!','#','"','%','$','&','(',')','+','*','-',',','/','.',';',
             ':','=','<','?','>','@','[',']','_','^','`','{','}','~', "'"]

    punctuation_dict = defaultdict(int)
    period_count = 0
    total_words = 0
    punct_count = 0

    # sentence count
    for i in range(1, len(character_list)):
        # if letter followed by space or punct, then word count +=1
        if ((character_list[i] == " " or str(character_list[i]) in punct) and
                str(character_list[i - 1]) in string.ascii_letters):
            total_words += 1
        # count periods
        if character_list[i] == ".":
            period_count += 1
        if character_list[i] in punct:
            punct_count += 1
            punctuation_dict[character_list[i]] += 1

    avg_sent_size = (total_words / period_count)
    # put together output, bar delimited
    pos_file.write(str(total_words) + "|")
    pos_file.write(str(avg_sent_size) + "|")

    for p in punct:
        s = ""
        if p in punctuation_dict:
            s = s + str(punctuation_dict[p] / punct_count) + "|"  # ratio of punct of all punct
        else:
            s = s + str(0) + "|"  # 0 if unused
        pos_file.write(s)


def get_sentiment(content):
    content = content.replace('\n', '')
    content = content.replace('\r', '')
    # tokenize sentences
    sentences = tokenize.sent_tokenize(content)

    # get author and title now that content is split by sentence
    sid = SentimentIntensityAnalyzer()
    booksent = []
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        ssarray = [ss['neg'], ss['neu'], ss['pos'], ss['compound']]
        booksent.append(ssarray)
    valuearray = np.array(booksent)
    # mean negative, neutral, positive, compound score for all lines in book
    values = np.mean(valuearray, axis=0)
    return values, booksent


def get_author(book_title):
    book_list = {'Agatha Christie': ['AndThenThereWereNone',
                                     'DestinationUnknown',
                                     'ElephantsCanRemember'],
                 'Iris Murdoch': ['TheSandcastle',
                                  'TheBlackPrince',
                                  'JacksonsDilemma'],
                 'P.D. James': ['CoverHerFace',
                                'DevicesAndDesires',
                                'DeathComesToPemberley']
                 }

    for author, books in book_list.items():
        if book_title in books and books[0] == book_title:
            return author, 1
        if book_title in books and books[1] == book_title:
            return author, 2
        if book_title in books and books[2] == book_title:
            return author, 3


def pos_tagging(content, pos_file):
    parts = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ",
             "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS",
             "NNS", "PDT", "PRP", "PRP$", "RB", "RBR",
             "RBS", "RP", "VB", "VBD", "VBG", "VBP",
             "VBN", "WDT", "VBZ", "WRB", "WP$", "WP"]
    # tokenize first
    text = nltk.word_tokenize(content)
    results = nltk.pos_tag(text)

    # dict of {POS: count}
    results_dict = defaultdict(int)
    counter = 0
    for tag in results:
        token = tag[0]
        pos = tag[1]
        counter += 1
        results_dict[pos] += 1

    # write to file
    for part_of_sp in parts:
        s = ""
        if part_of_sp in results_dict:
            # percent of POS
            s = s + str(results_dict[part_of_sp] / float(counter)) + "|"
        else:
            s = s + str(0) + "|"  # 0 if unused
        pos_file.write(s)


def preprocessing():
    '''
    read file as a list of words
    get punctuation string for later feature extraction
    save local wordcount dict???
    save global word dict after finished looping through docs???
    '''

    pos_file = open("data/processed/output_POS.txt", 'a')

    # check avg sent size
    pos_file.write("Author|Title|book_order|total_words|avg_sentence_size|"
                         + "!|#|\"|%|$|&|(|)|+|*|-|,|/|.|;|:|=|<|?|>|"
                         + "@|[|]|_|^|`|{|}|~|\'|neg|neu|pos|compound|"
                         + "CC|CD|DT|EX|FW|IN|JJ|JJR|JJS|"
                         + "LS|MD|NN|NNP|NNPS|NNS|PDT|PRP|PRP$|RB|RBR|"
                         + "RBS|RP|VB|VBD|VBG|VBP|VBN|WDT|VBZ|WRB|WP$|WP|")

    pos_file.write('\n')


    for book in os.listdir("data/interim"):
        book_file = str(book)
        book_name = re.sub(r'(James[0-9]_|Murdoch[0-9]_|Christie[0-9]_|\.txt)*', '', book_file)
        title = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", book_name)
        author, order = get_author(book_name)
        pos_file.write(author + "|" + title + "|" )
        pos_file.write(str(order) + "|")

        with open("data/interim/" + book_file, 'r') as f:
            content = f.read().rstrip('\n')

        punct_and_words(content, pos_file)
        sentiment_values, _ = get_sentiment(content)
        neg = sentiment_values[0]
        neu = sentiment_values[1]
        pos = sentiment_values[2]
        compound = sentiment_values[3]
        pos_file.write(str(neg) + "|"
                             + str(neu) + "|"
                             + str(pos) + "|"
                             + str(compound) + "|")


        pos_tagging(content, pos_file)
        pos_file.write('\n')
        print(f'Done processing: {title}')
        f.close()
    pos_file.close()

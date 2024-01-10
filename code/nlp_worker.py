"""
Mehran Ali Banka - Sep 2023
----------------------------

This file contains various NLP related tasks
used by the code. Uses packages like nltk

"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import math
import string
# Word net has been heavily used for content expansion
# and similarity computation
# nltk.download('wordnet')
from nltk.corpus import wordnet

# max words in wordnet
max_w = 155000
translation_table = str.maketrans("", "", string.punctuation)
stop_words = set(stopwords.words('english'))
ic_cache = {}
synset_cache = {}
lcs_cache = {}

def extract_sentences_with_ids(paragraph):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Assign unique IDs to each sentence
    sentences_with_ids = {i : sentence for i, sentence in enumerate(sentences)}

    return sentences_with_ids

# returns processed sentences with ids
def extract_processed_sentences_with_ids(paragraph):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    
    processed_sentences = []

    for sentence in sentences:
        processed_sentences.append(process_sentence(sentence))

    # Assign unique IDs to each sentence
    sentences_with_ids = {i : sentence for i, sentence in enumerate(processed_sentences)}

    return sentences_with_ids

# processes sentences - stop word removal, spell check
def process_sentence(sentence):

    # Remove punctuations
    sentence = sentence.translate(translation_table)
    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Remove stop words
    words_without_stopwords = [word for word in words if word.lower() not in stop_words]

    # Perform spelling correction using TextBlob
    #corrected_words = [str(TextBlob(word).correct()) for word in words_without_stopwords]  

    # Join the corrected words to form the processed sentence
    processed_sentence = ' '.join(words_without_stopwords)  # corrected_words

    return processed_sentence

# Create the dictionary of sentence id -> set of words in the sentence
# Each word in the individual sentences is replaced by its root word
# from WordNet
def create_word_set(sentence_dict):

    # Create a WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Create a new dictionary with IDs as keys and sets of lemmatized words as values
    lemmatized_word_sets_dict = {}

    for id, sentence in sentence_dict.items():
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
    
        # Lemmatize each word and store in a list
        lemmatized_words_list = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(word)) for word in words]
    
        # Assign the set of lemmatized words to the corresponding ID in the new dictionary
        lemmatized_word_sets_dict[id] = lemmatized_words_list

    return lemmatized_word_sets_dict

# Function to get the POS (Part of Speech) tag for WordNet lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}
    return tag_dict.get(tag, wordnet.NOUN)

def get_information_content(word):
    if(word in ic_cache): return ic_cache[word]
    # no of synonyms of the word
    synset_m = len(wordnet.synsets(word))
    ic = 1 - (math.log(synset_m + 1)/math.log(max_w))
    ic_cache[word] = ic
    return ic

# Get least common subsumer of two words
def least_common_subsumer(word1, word2):
    synsets1 = None 
    synsets2 = None
    if(str(word1) in synset_cache): synsets1 = synset_cache[word1]
    else: synsets1 = wordnet.synsets(word1)
    if(str(word2) in synset_cache): synsets2 = synset_cache[word2]
    else: synsets2 = wordnet.synsets(word2)
    
    synset_cache[word1] = synsets1
    synset_cache[word2] = synsets2
    
    if(synsets1 is None or synsets2 is None): return None

    # Find common hypernyms
    common_hypernyms = []
    key1 = str(synsets1) + str(synsets2)
    key2 = str(synsets2) + str(synsets1)
    if(key1 in lcs_cache): return lcs_cache[key1]
    if(key2 in lcs_cache): return lcs_cache[key2]
 
    for synset1 in synsets1:
        for synset2 in synsets2:
            common_hypernyms.extend(synset1.lowest_common_hypernyms(synset2))

    # Find the most specific common hypernym
    if common_hypernyms:
        lcs = max(common_hypernyms, key=lambda x: x.max_depth())
        lcs_cache[key1] = lcs
        return lcs
    else:
        return None


# Content word expansion algorithm when the word is not in word set
def content_word_expansion_score(word, word_list):
    # information content of the main word
    ic_m = get_information_content(word)
    max_sim_score = 0
    max_idx = 0
    for idx,word_n in enumerate(word_list):
        ic_n = get_information_content(word_n)
        # get the least common subsumer (LCS)
        lcs = least_common_subsumer(word,word_n)
        if lcs is None:
            continue
        lcs = lcs.lemmas()[0].name() 
        # get information content of LCS
        ic_lcs = get_information_content(lcs)
        sim_score = 2*ic_lcs/(ic_m + ic_n)
        if(sim_score > max_sim_score):
            max_sim_score = sim_score
            max_idx = idx

    return max_sim_score, max_idx    


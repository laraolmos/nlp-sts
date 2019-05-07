# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import re
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from nltk import pos_tag
# http://www.nltk.org/howto/wordnet.html
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer



# Segment text in phrases or words

class TextTokenizer():

	def sentences(self, input_text):
		#spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
		english_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		return english_tokenizer.tokenize(input_text)

	def words(self, input_text):
		return word_tokenize(input_text)

	def tokenize_corpus(self, corpus):
		return [self.words(text) for text in corpus]


# Process after tokenization

class TextNormalizer():

	def __init__(self):
		#self.stemmer = SnowballStemmer('spanish')
		self.stemmer = SnowballStemmer('english')

	def _remove_puntuation(self, word):
		regular_expr = re.compile('\r|\n|\t|\(|\)|\[|\]|:|\.|\,|\;|"|”|…|»|“|/|\'|\?|\¿|\!|\¡|`|\%|\.\.\.|-|—|=|–|―|@|#')
		word_processed = re.sub(regular_expr, '', word)
		#regex = re.compile('[^A-Za-z ]')
		#result = regex.sub('', word_processed)
		return word_processed

	def _filter_words(self, sentence):
		return [token for token in sentence if token not in [' ', ''] and len(token) < 10 ]

	# not used
	def _remove_numbers(self, word):
		num_expr = re.compile('[0-9]+|[0-9]*[,.][0-9]+')
		word_processed = re.sub(num_expr, '', word)
		return word_processed

	def _remove_stopwords(self, sentence):
		return [token for token in sentence if token not in stopwords.words('english')]

	def _stemming(self, token):
		return self.stemmer.stem(token)

	# normalization pipeline
	def normalize_token_list(self, token_list):
		processed_token_list = []
		for word in token_list:
			word_processed = word.lower().strip()
			word_processed = self._remove_puntuation(word_processed)
			processed_token_list.append(word_processed)
		processed_token_list = self._filter_words(processed_token_list)
		return processed_token_list

	def normalize_corpus(self, corpus):
		return [self.normalize_token_list(token_list) for token_list in corpus]


class PreprocessWordNet():

	def _wordnet_postag(self, nltk_tag):
		if 'NN' in nltk_tag:
			return wn.NOUN # 'n'
		if 'VB' in nltk_tag:
			return wn.VERB # 'v'
		if 'RB' in nltk_tag:
			return wn.ADV # 'r'
		if 'JJ' in nltk_tag:
			return wn.ADJ # 'a'
		return None

	def token_list_synsets(self, token_list):
		tags = pos_tag(token_list)
		synsets_list = []
		valid_tokens = []
		wn_postags = []
		for token in tags:
			wn_valid = self._wordnet_postag(token[1])
			if wn_valid:
				wordnet_token = WordNetLemmatizer().lemmatize(token[0], pos=wn_valid)
				synset = wn.synsets(wordnet_token, pos=wn_valid)
				if synset:
					valid_tokens.append(wordnet_token)
					synsets_list.append(synset[0])
					wn_postags.append((wordnet_token, wn_valid))
		return synsets_list, valid_tokens, wn_postags

if __name__ == '__main__':
	example_tokens = ['a', 'young', 'child', 'is', 'riding', 'a', 'horse']
	preprocessor = PreprocessWordNet()
	synsets_list, valid_tokens = preprocessor.token_list_synsets(example_tokens)
	example_str = str(example_tokens) + '\n' + str(valid_tokens) + '\n' + str(synsets_list)
	print(example_str)
	#export_to_txt('example_wordnet.txt', example_str)
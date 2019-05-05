# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import os

from settings import *
from extractor import *
from preprocessor import *

from export import export_to_txt

import numpy as np
import itertools
from wordcloud import WordCloud
import matplotlib.pyplot as plt


from nltk import pos_tag



class STS_Corpus():

	def __init__(self, input_dir=CORPUS_PATH):
		self.input_path = input_dir
		self.text_extractor = STSExtractor()
		self.text_tokenizer = TextTokenizer()
		self.text_normalizer = TextNormalizer()
		self.examples, self.processed_text, self.partitions_normalized, self.terms, self.sts_dataset = self._build()

	def _build(self):
		dir_list_files = sorted(os.listdir(self.input_path))
		examples, pearsons = {}, {}
		processed_text, sts_dataset = [], []
		partitions, partitions_normalized = {}, {}
		terms = {}
		for file_name in dir_list_files:
			examples[file_name] = self.text_extractor.extract_file_examples(self.input_path + file_name)
			sts_dataset += [(self.text_normalizer.normalize_corpus(
									self.text_tokenizer.tokenize_corpus(self.text_extractor.get_texts(example))), 
								self.text_extractor.get_pearson(example), 
								self.text_extractor.get_category(example)) for example in examples[file_name]]
			partitions = self.text_extractor.get_text_partitions(examples[file_name], dictionary_partition=partitions)
		for category in partitions.keys():
			partitions_normalized[category] = self.text_normalizer.normalize_corpus(
				self.text_tokenizer.tokenize_corpus(partitions[category]))
			terms[category] = list(itertools.chain(*partitions_normalized[category]))
		processed_text += list(partitions_normalized.values())
		return examples, processed_text, partitions_normalized, terms, sts_dataset

	def _generate_word_cloud(self, text, name):		
		word_cloud = WordCloud().generate(text)
		# Display the generated image:
		plt.imshow(word_cloud)
		plt.axis("off")
		#plt.show()
		plt.savefig(OUTPUT_DIR + 'wordcloud' + name + '.png')

	def description(self):
		lengths = {}
		postags = {}
		count_postags = {}
		for category in self.partitions_normalized.keys():
			lengths[category] = np.mean([len(example_tokens) for example_tokens in self.partitions_normalized[category]])
			postags[category] = [nltk.pos_tag(example_tokens) for example_tokens in self.partitions_normalized[category]]
			postags[category] = list(itertools.chain(*postags[category]))
			postags[category] = [tag[1] for tag in postags[category]]
			count_postags[category] = sorted([(postags[category].count(tag), tag) for tag in set(postags[category])])
			category_text = ' '.join(self.terms[category])
			self._generate_word_cloud(category_text, category)
		export_to_txt('lengths.txt', str(lengths))
		export_to_txt('tags.txt', str(count_postags))

	def words(self):
		return list(itertools.chain(*self.terms.values()))

	def sts_parts(self, category='all'):
		sentences, pearsons, = [], []
		for example in self.sts_dataset:
			if (category != 'all' and category == example[2]) or category == 'all':
				if example[0][0] and example[0][1]:
					sentences.append(example[0])
					pearsons.append(float(example[1]))
		return sentences, pearsons

if __name__ == '__main__':
	corpus = STS_Corpus()
	corpus.description()
	export_to_txt('processed_sts.txt', str(corpus.sts_dataset))


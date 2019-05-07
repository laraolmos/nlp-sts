# -*- coding: utf-8 -*-

__author__ = 'Lara Olmos Camarena'

from lexicalsimilarity import LexicalSimilarity
from corpus import STS_Corpus

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math


class TextSimilarity():

	def __init__(self, corpus=STS_Corpus(), similarity_function='path_similarity', ic=None):
		if similarity_function:
			self.lexicalsimilarity = LexicalSimilarity(corpus, ic=ic)
			self.similarity_function = similarity_function

	def _union_set(self, token_list1, token_list2):
		if token_list1 and token_list2:
			union_token_list = list(set(token_list1 + token_list2))
			union_token_list.sort()
			return union_token_list

	def _lexical_semantic_vector(self, union_token_list, token_list):
		semantic_vector = np.zeros(len(union_token_list))
		for i in range(0, len(union_token_list)):
			partial_similarity = np.zeros(len(token_list))
			for j in range(0, len(token_list)):
				partial_similarity[j] = self.lexicalsimilarity.choose_similarity(
					self.similarity_function, union_token_list[i], token_list[j])
			semantic_vector[i] = np.nanmax(partial_similarity)
			if math.isnan(semantic_vector[i]):
				semantic_vector[i] = 0
		return semantic_vector

	def _one_hot_semantic(self, union_token_list, token_list):
		semantic_vector = np.zeros(len(union_token_list))
		for i in range(0, len(union_token_list)):
			if union_token_list[i] in token_list:
				semantic_vector[i] = 1
		return semantic_vector

	def cosine_sentence_similarity(self, token_list1, token_list2):
		union_token_list = self._union_set(token_list1, token_list2)
		if union_token_list:
			vector1 = self._lexical_semantic_vector(union_token_list, token_list1)
			vector2 = self._lexical_semantic_vector(union_token_list, token_list2)
			return cosine_similarity(vector1.reshape((1,len(union_token_list))), vector2.reshape((1, len(union_token_list))))[0][0]
		return 0

	def word_overlap_similarity(self, token_list1, token_list2):
		union_token_list = self._union_set(token_list1, token_list2)
		if union_token_list:
			set1 = set(token_list1)
			set2 = set(token_list2)
			intersection = set1.intersection(set2)
			return 2*len(intersection) / (len(set1) + len(set2))
		return 0

	def jaccard_similarity(self, token_list1, token_list2):
		union_token_list = self._union_set(token_list1, token_list2)
		if union_token_list:
			set1 = set(token_list1)
			set2 = set(token_list2)
			intersection = set1.intersection(set2)		
			return len(intersection) / (len(set1) + len(set2) - len(intersection))
		return 0


if __name__ == '__main__':

	example1 = ['a', 'man', 'with', 'a', 'hard', 'hat', 'is', 'dancing']
	example2 = ['a', 'man', 'wearing', 'a', 'hard', 'hat', 'is', 'dancing']
	example3 = ['a', 'man', 'wearing', 'a', 'soft', 'hat', 'is', 'jumping']

	textsimilarity = TextSimilarity()
	print(textsimilarity.cosine_sentence_similarity(example1, example2))
	print(textsimilarity.cosine_sentence_similarity(example1, example3))

	print(str(textsimilarity.lexicalsimilarity.path_similarity('wearing', 'is')))

	print(textsimilarity.word_overlap_similarity(example1, example2))
	print(textsimilarity.word_overlap_similarity(example1, example3))

	print(textsimilarity.jaccard_similarity(example1, example2))
	print(textsimilarity.jaccard_similarity(example1, example3))
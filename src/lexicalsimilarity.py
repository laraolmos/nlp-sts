# -*- coding: utf-8 -*-

__author__ = 'Lara Olmos Camarena'

from preprocessor import PreprocessWordNet
from corpus import STS_Corpus
from export import export_to_txt

# http://www.nltk.org/howto/wordnet.html
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError


class LexicalSimilarity():

	def __init__(self, corpus, ic=None):
		self.preprocessor = PreprocessWordNet()
		if not ic:
			self.ic = wn.ic(corpus, False, 0.0)
		else:
			self.ic = ic

	def _get_synsets(self, token1, token2):
		synsets1,_,pos_tags1 = self.preprocessor.token_list_synsets([token1])
		synsets2,_,pos_tags2 = self.preprocessor.token_list_synsets([token2])
		if synsets1 and synsets2 and pos_tags1 and pos_tags2:
			pos_tag_valid1 = pos_tags1[0][1]
			pos_tag_valid2 = pos_tags2[0][1]
			if pos_tag_valid1 == pos_tag_valid2:
				if pos_tag_valid1 == wn.ADJ:
					posible_synset1 = [result for result in synsets1 if '.a.' in str(synsets1)]
					posible_synset2 = [result for result in synsets2 if '.a.' in str(synsets2)]
					if posible_synset1 and posible_synset2:
						return posible_synset1[0], posible_synset2[0]
					posible_synset1 = [result for result in synsets1 if '.s.' in str(synsets1)]
					posible_synset2 = [result for result in synsets2 if '.s.' in str(synsets2)]
					if posible_synset1 and posible_synset2:
						return posible_synset1[0], posible_synset2[0]
					return None,None
				return synsets1[0], synsets2[0]
		return None,None

	def path_similarity(self, token1, token2):
		first_synset, sec_synset = self._get_synsets(token1,token2)
		if first_synset and sec_synset:
			return first_synset.path_similarity(sec_synset)
		return 0

	def lch_similarity(self, token1, token2):
		first_synset, sec_synset = self._get_synsets(token1,token2)
		if first_synset and sec_synset:
			return first_synset.lch_similarity(sec_synset)
		return 0

	def wup_similarity(self, token1, token2):
		first_synset, sec_synset = self._get_synsets(token1,token2)
		if first_synset and sec_synset:
			return first_synset.wup_similarity(sec_synset)
		return 0

	def res_similarity(self, token1, token2):
		first_synset, sec_synset = self._get_synsets(token1,token2)
		if first_synset and sec_synset:
			return first_synset.res_similarity(sec_synset, self.ic)
		return 0

	def jcn_similarity(self, token1, token2):
		first_synset, sec_synset = self._get_synsets(token1,token2)
		if first_synset and sec_synset:
			return first_synset.jcn_similarity(sec_synset, self.ic)
		return 0

	def lin_similarity(self, token1, token2):
		first_synset, sec_synset = self._get_synsets(token1,token2)
		if first_synset and sec_synset:
			return first_synset.lin_similarity(sec_synset, self.ic)
		return 0

	def choose_similarity(self, similarity_name, token1, token2):
		try:
			if similarity_name == 'path_similarity':
				return self.path_similarity(token1, token2)
			if similarity_name == 'lch_similarity':
				return self.lch_similarity(token1, token2)
			if similarity_name == 'wup_similarity':
				return self.wup_similarity(token1, token2)
			if similarity_name == 'res_similarity':
				return self.res_similarity(token1, token2)
			if similarity_name == 'jcn_similarity':
				return self.jcn_similarity(token1, token2)
			if similarity_name == 'lin_similarity':
				return self.lin_similarity(token1, token2)
		except WordNetError:
			print('Error: ' + token1 + ' ' + token2)
			return 0


if __name__ == '__main__':

	stscorpus = STS_Corpus()
	lexicalsim = LexicalSimilarity(stscorpus)

	#lexical_similarity = 'path_similarity("boy","child"): ' + str(lexicalsim.path_similarity('boy', 'child')) + '\n' + 'lch_similarity("boy","child"): ' + str(lexicalsim.lch_similarity('boy', 'child')) + '\n' + 'wup_similarity("boy","child"): ' + str(lexicalsim.wup_similarity('boy', 'child')) + '\n' + 'res_similarity("boy","child"): ' + str(lexicalsim.res_similarity('boy', 'child')) + '\n' + 'jcn_similarity("boy","child"): ' + str(lexicalsim.jcn_similarity('boy', 'child')) + '\n' + 'lin_similarity("boy","child"): ' + str(lexicalsim.lin_similarity('boy', 'child'))
	#lexical_similarity = 'path_similarity("boy","man"): ' + str(lexicalsim.path_similarity('boy', 'man')) + '\n' + 'lch_similarity("boy","man"): ' + str(lexicalsim.lch_similarity('boy', 'man')) + '\n' + 'wup_similarity("boy","man"): ' + str(lexicalsim.wup_similarity('boy', 'man')) + '\n' + 'res_similarity("boy","man"): ' + str(lexicalsim.res_similarity('boy', 'man')) + '\n' + 'jcn_similarity("boy","man"): ' + str(lexicalsim.jcn_similarity('boy', 'man')) + '\n' + 'lin_similarity("boy","man"): ' + str(lexicalsim.lin_similarity('boy', 'man'))
	#lexical_similarity = 'path_similarity("boy","adult"): ' + str(lexicalsim.path_similarity('boy', 'adult')) + '\n' + 'lch_similarity("boy","adult"): ' + str(lexicalsim.lch_similarity('boy', 'adult')) + '\n' + 'wup_similarity("boy","adult"): ' + str(lexicalsim.wup_similarity('boy', 'adult')) + '\n' + 'res_similarity("boy","adult"): ' + str(lexicalsim.res_similarity('boy', 'adult')) + '\n' + 'jcn_similarity("boy","adult"): ' + str(lexicalsim.jcn_similarity('boy', 'adult')) + '\n' + 'lin_similarity("boy","adult"): ' + str(lexicalsim.lin_similarity('boy', 'adult'))
	#lexical_similarity = 'path_similarity("boy","girl"): ' + str(lexicalsim.path_similarity('boy', 'girl')) + '\n' + 'lch_similarity("boy","girl"): ' + str(lexicalsim.lch_similarity('boy', 'girl')) + '\n' + 'wup_similarity("boy","girl"): ' + str(lexicalsim.wup_similarity('boy', 'girl')) + '\n' + 'res_similarity("boy","girl"): ' + str(lexicalsim.res_similarity('boy', 'girl')) + '\n' + 'jcn_similarity("boy","girl"): ' + str(lexicalsim.jcn_similarity('boy', 'girl')) + '\n' + 'lin_similarity("boy","girl"): ' + str(lexicalsim.lin_similarity('boy', 'girl'))
	#export_to_txt('lexical_similarity.txt', lexical_similarity)
	
	print(lexicalsim.choose_similarity('lch_similarity', 'green', 'large'))
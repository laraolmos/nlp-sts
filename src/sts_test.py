# -*- coding: utf-8 -*-

__author__ = 'Lara Olmos Camarena'


from corpus import STS_Corpus
from textsimilarity import TextSimilarity
from preprocessor import PreprocessWordNet
from export import export_to_json

# http://www.nltk.org/howto/wordnet.html
from nltk.corpus import wordnet as wn
from nltk.corpus import brown

import numpy as np


def test_text_similarities(dataset, textsimilarity):
	similarity_measures = []
	for item in dataset:
		similarity_measures.append(textsimilarity.cosine_sentence_similarity(item[0], item[1]))
	return similarity_measures

def test_word_overlap(dataset, textsimilarity):
	similarity_measures = []
	for item in dataset:
		similarity_measures.append(textsimilarity.word_overlap_similarity(item[0], item[1]))
	return similarity_measures

def test_jaccard(dataset, textsimilarity):
	similarity_measures = []
	for item in dataset:
		similarity_measures.append(textsimilarity.jaccard_similarity(item[0], item[1]))
	return similarity_measures

def pearson_coefficient(similarity_measures, sts_pearsons):
	return np.corrcoef(np.array(similarity_measures), np.array(sts_pearsons))

def corpus_wn_valid(sentences):
	sentences_valid = []
	pwn = PreprocessWordNet()
	for pair_sentences in sentences:
		synsets_list1, valid_tokens1, pos_tags1 = pwn.token_list_synsets(pair_sentences[0])
		synsets_list2, valid_tokens2, pos_tags2 = pwn.token_list_synsets(pair_sentences[1])
		sentences_valid.append([valid_tokens1, valid_tokens2])
	return sentences_valid

def test_lexical_similarity(similarity_name, corpus, sentences, dataset_ic, pearson, category, removal):
	print(similarity_name)	
	similarity_obj = TextSimilarity(corpus=corpus, similarity_function=similarity_name, ic=dataset_ic)
	similarity_measures = test_text_similarities(sentences, similarity_obj)
	coeficient_result = pearson_coefficient(similarity_measures, pearson)[0][1]
	print(coeficient_result)
	results = {'dataset': category, 'removal': removal, 'similarity': similarity_name, 'results': similarity_measures, 'coeficient': coeficient_result}
	export_to_json(similarity_name + '_' + category + '_' + removal +'.json', results)

def test_overlap_similarity(corpus, sentences, pearson, category, removal, similarity_name='word_overlap_aligned'):
	print(similarity_name)	
	similarity_obj = TextSimilarity(corpus=corpus, similarity_function=None, ic=None)
	similarity_measures = test_word_overlap(sentences, similarity_obj)
	coeficient_result = pearson_coefficient(similarity_measures, pearson)[0][1]
	print(coeficient_result)
	results = {'dataset': category, 'removal': removal, 'similarity': similarity_name, 'results': similarity_measures, 'coeficient': coeficient_result}
	export_to_json(similarity_name + '_' + category + '_' + removal + '.json', results)

def test_jaccard_similarity(corpus, sentences, pearson, category, removal, similarity_name='jaccard'):
	print(similarity_name)	
	similarity_obj = TextSimilarity(corpus=corpus, similarity_function=None, ic=None)
	similarity_measures = test_jaccard(sentences, similarity_obj)
	coeficient_result = pearson_coefficient(similarity_measures, pearson)[0][1]
	print(coeficient_result)
	results = {'dataset': category, 'removal': removal, 'similarity': similarity_name, 'results': similarity_measures, 'coeficient': coeficient_result}
	export_to_json(similarity_name + '_' + category + '_' + removal +'.json', results)


if __name__ == '__main__':

	print('Load corpus and train IC')
	corpus = STS_Corpus()
	sts_ic = wn.ic(corpus, False, 0.0)
	# brown_ic = wn.ic(brown, False, 0.0)

	all_categories = ['all', 'main-captions', 'main-forums', 'main-news']
	category = all_categories[0]

	sentences, pearson = corpus.sts_parts(category=category)
	#sentences = corpus_wn_valid(sentences)
	removal = 'no'

	#dataset = {'texts': sentences, 'pearsons': pearson, 'category': category}
	#export_to_json('sts_' + category + '_' + removal + '.json', dataset)

	test_lexical_similarity('path_similarity', corpus, sentences, sts_ic, pearson, category, removal)
	#test_lexical_similarity('lch_similarity', corpus, sentences, sts_ic, pearson, category, removal)
	#test_lexical_similarity('wup_similarity', corpus, sentences, sts_ic, pearson, category, removal)
	#test_lexical_similarity('res_similarity', corpus, sentences, sts_ic, pearson, category, removal)
	#test_lexical_similarity('jcn_similarity', corpus, sentences, sts_ic, pearson, category, removal)
	#test_lexical_similarity('lin_similarity', corpus, sentences, sts_ic, pearson, category, removal)
	#test_overlap_similarity(corpus, sentences, pearson, category, removal)
	#test_jaccard_similarity(corpus, sentences, pearson, category, removal)

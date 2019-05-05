# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import codecs
from settings import OUTPUT_DIR

import json


def export_to_txt(file_name, text):
	with codecs.open(OUTPUT_DIR + file_name, 'w', encoding='utf-8') as f:
		f.write(text)

def export_dictionary_to_txt(filename, dictionary):
	str_list = [str(key) + ' -> ' + str(dictionary[key]) for key in sorted(dictionary.keys())]
	export_to_txt(filename, '\n'.join(str_list))

def export_to_json(file_name, data):
	with codecs.open(OUTPUT_DIR + file_name, 'w', encoding='utf-8') as f:
		json.dump(data, f)

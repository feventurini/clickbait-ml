import numpy as np
from collections import Counter
import re
from keras import callbacks
from keras import losses
import pickle

def pickle_dump(filename, obj):
	with open(filename, 'wb') as dump:
	    pickle.dump(obj, dump)

def pickle_load(filename):
	with open(filename, 'rb') as dump:
	    obj = pickle.load(dump)
	return obj

def gen_vocab(infile, max_size=10000, save=False):
	counter = Counter()
	with open(infile) as f:
		for sentence in f:
			counter.update(sentence.split())

	vocab = counter.most_common(max_size - 3)
	word2id = { e[0]:i for e,i in zip(vocab,range(4,len(vocab)+4)) }
	# 0 is used for masking
	word2id['<s>'] = 1
	word2id['</s>'] = 2
	word2id['<unk>'] = 3
	id2word = { i:word for word,i in word2id.items() }

	if save:
		pickle_dump('../data/word2id.p', word2id)

	return word2id, id2word

def load_embeddings(file, word2id, units):
	embedding_matrix = np.random.uniform(-0.05, 0.05, (len(word2id)+1, units))
	with open(file) as f:
		for line in f:
			tokens = line.split()
			word, values = tokens[0], tokens[1:]
			if word in word2id:
				embedding_matrix[word2id[word],:] = np.asarray(values)
	return embedding_matrix

def multi_array_shuffle(tuple):
    rng_state = np.random.get_state()
    for array in tuple:
	    np.random.shuffle(array)
	    np.random.set_state(rng_state)

def perplexity(y_true, y_pred):
	## TODO
	pass

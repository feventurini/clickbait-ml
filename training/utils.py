import numpy as np
from collections import Counter
import re
from keras import callbacks
from keras import losses

def gen_vocab(infile, max_size=10000):
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

def generate_sentence(model, word2id, id2word, tbptt_step, maxlen=20, time_distributed=False):
	sentence = ''
	likelihood = 1.
	i = 0
	start = np.zeros((1, tbptt_step))

	if time_distributed: start[0, 0] = word2id['<s>']
	else: start[0, -1] = word2id['<s>']

	next_word = -1
	while next_word != word2id['</s>'] and len(sentence.split()) < maxlen:
		i += 1
		out = model.predict(start)
		probs = out[i % tbptt_step] if time_distributed else out[0]

		next_word = multinomial_sample(probs)
		while next_word == word2id['<unk>']:
			next_word = multinomial_sample(probs)

		likelihood *= probs[next_word]
		if next_word != word2id['</s>']:
			sentence += ' {}'.format(id2word[next_word])

		if time_distributed and i < tbptt_step:
			start[0,i] = next_word
		else:
			start = np.roll(start,-1,axis=1)
			start[0,-1] = next_word

	return re.sub(r'(\d)\s+(\d)', r'\1\2', sentence.strip()).title(), likelihood

## callback wrappers
class SentenceGeneratorCB(callbacks.Callback):
	def __init__(self, word2id, id2word, tbptt_step, time_distributed=True):
		self.word2id = word2id
		self.id2word = id2word
		self.tbptt_step = tbptt_step
		self.time_distributed = time_distributed

	def on_epoch_end(self, epoch, logs=None):
		print('\nSampling sentences...')
		for i in range(5):
			sentence, prob = generate_sentence(self.model, self.word2id, self.id2word, 
				self.tbptt_step, time_distributed=self.time_distributed)
			print(sentence)
			print('Probability: {}'.format(prob))
		print()		

class ComputePerplexityCB(callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		X_dev, y_dev = self.validation_data[:2]
		print('Perplexity on dev set: {}'.format(perplexity(y_dev, self.model.predict(X_dev))))

def multinomial_sample(p, k=1):
	## efficient O(log n) multinomial sampling
    cdf = np.cumsum(p.astype(float) / sum(p))
    rs = np.random.random(k)
    return np.searchsorted(cdf, rs)[0]

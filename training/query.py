from keras import models
import sys
import utils
import nltk
import string
import numpy as np
import re

to_replace = [" 'll", " 've", " n't", " 's", " 're"]

def multinomial_sample(p, k=1):
	## efficient O(log n) multinomial sampling
    cdf = np.cumsum(p.astype(float) / sum(p))
    rs = np.random.random(k)
    return np.searchsorted(cdf, rs)[0]

class SentenceGenerator(object):
	"""docstring for SentenceGenerator"""
	def __init__(self, model, word2id, id2word=None, tbptt_step=None):
		super(SentenceGenerator, self).__init__()
		self.model = model
		self.word2id = word2id
		self.id2word = id2word if id2word else { i:w for (w,i) in word2id.items() }
		self.tbptt_step = tbptt_step if tbptt_step else model.layers[0].input_length
		self.old_sent = None

	def __preprocess_start(self, start):
		if start == '':
			return []

		tokens = nltk.word_tokenize(start)
		result = []
		for t in tokens:
			if all(map(lambda i: i in string.punctuation, t)):
				continue
			try:
				int(t)
				num = True
			except:
				num = False
			if num: result += list(t)
			else: result.append(t.lower())

		return result

	def __postprocess_out(self, out):
		result = out.strip()
		for i in range(2): 
			result = re.sub(r'(\d)\s+(\d)', r'\1\2', result)

		for i in to_replace:
			result = result.replace(i, i.strip())

		return string.capwords(result)


	def __generate_sentence(self, start, maxlen):
		start_tokens = self.__preprocess_start(start)
		sentence = ' '.join(start_tokens)
		likelihood = 1.

		start_tokens.insert(0, '<s>')
		
		for i,t in enumerate(start_tokens[-5:]):
			input_tokens = np.zeros((1, self.tbptt_step))

			input_tokens[0, i] = self.word2id[t]

		next_word = -1
		while next_word != self.word2id['</s>'] and len(sentence.split()) < maxlen:
			out = self.model.predict(input_tokens)
			probs = out[0]

			next_word = multinomial_sample(probs)
			while next_word == self.word2id['<unk>']:
				next_word = multinomial_sample(probs)

			likelihood *= probs[next_word]
			if next_word != self.word2id['</s>']:
				sentence += ' {}'.format(self.id2word[next_word])

			input_tokens = np.roll(input_tokens,-1,axis=1)
			input_tokens[0,-1] = next_word

		return self.__postprocess_out(sentence), likelihood

	def __check_diversity(self, s):

		if len(s.split())<5:
			return False

		new_sent = set(s.split())
		if self.old_sent and len(new_sent.intersection(self.old_sent)) > 2:
			return False
		else:
			self.old_sent = new_sent
			return True

	def query(self, start='', n=10, maxlen=20):
		diversity = False

		while not diversity:
			options = []
			likelihoods = []
			for i in range(n):
				o, l = self.__generate_sentence(start, maxlen)
				options.append(o)
				likelihoods.append(l)

			likelihoods, options = (list(j) for j in zip(*sorted(zip(likelihoods, options), reverse=True)))

			i = 0
			while i<5 and not diversity:
				result_s = options[i]
				result_l = likelihoods[i]

				diversity = self.__check_diversity(result_s)
				i += 1

		return result_s, result_l

if __name__ == '__main__':
	model = models.load_model(sys.argv[1])
	word2id = utils.pickle_load(sys.argv[2])

	sengen = SentenceGenerator(model, word2id)

	while True:
		try:
			start = input('Insert beginning of the title (or leave empty): ')
			if not start: start = '' 
			print(sengen.query(start)[0])
		except EOFError:
			print('Thanks, bye!')

import nltk
import string

dataset_in = '../data/buzzfeed_dataset.txt'
dataset_out = '../data/preprocessed_buzzfeed_dataset.txt'

seen = set()
with open(dataset_in) as infile:
	with open(dataset_out, 'w+') as outfile:
		count = 0
		for s in infile:
			if s in seen:
				continue
			seen.add(s)
			tokens = nltk.word_tokenize(s)
			to_print = []
			for t in tokens:
				if all(map(lambda i: i in string.punctuation, t)):
					continue

				try:
					int(t)
					num = True
				except:
					num = False

				if num: to_print += list(t)
				else: to_print.append(t.lower())
				
			outfile.write(' '.join(to_print) + '\n')
			count += 1
			if not count % 1000:
				print('Processed {} sentences'.format(count))
				outfile.flush()



from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, LSTM, GRU, Dropout
from keras.preprocessing import sequence
from keras.losses import categorical_crossentropy
from keras import callbacks

import os
import pickle
import numpy as np
import utils
import query

## callback wrappers
class SentenceGeneratorCB(callbacks.Callback):
	def __init__(self, word2id, id2word=None, tbptt_step=None):
		self.word2id = word2id
		self.id2word = id2word
		self.tbptt_step = tbptt_step

	def on_train_begin(self, logs=None):
		self.sentence_generator = query.SentenceGenerator(self.model, self.word2id, id2word=self.id2word, tbptt_step=self.tbptt_step)

	def on_epoch_end(self, epoch, logs=None):
		print('\nSampling sentences...')
		for i in range(5):
			sentence, prob = self.sentence_generator.query(n=1)
			print(sentence)
			print('Probability: {}'.format(prob))
		print()		

class ComputePerplexityCB(callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		X_dev, y_dev = self.validation_data[:2]
		print('Perplexity on dev set: {}'.format(utils.perplexity(y_dev, self.model.predict(X_dev))))

dataset = '../data/preprocessed_buzzfeed_dataset.txt'
word2id, id2word = utils.gen_vocab(dataset, save=True)

tbptt_step = 10

with open(dataset) as f:
	sentences = []
	target_word = []
	for s in f:
		s = '<s> ' + s + ' </s>'
		tokens_orig = s.split()
		tokens = list(map(lambda w: w if w in word2id else '<unk>', tokens_orig))
		j = 0
		for i in range(len(tokens) - tbptt_step):
			if j==0:
				for j in range(1,tbptt_step):
					sentences.append(tokens[i: i + j])
					target_word.append(tokens[i + j])					
			sentences.append(tokens[i: i + tbptt_step])
			target_word.append(tokens[i + tbptt_step])

X = np.empty((len(sentences), tbptt_step))
y = np.empty((len(sentences), 1))

val_size = int(.1*len(X))

for i,s in enumerate(sentences):
	for j,word in enumerate(s):
		X[i,j] = word2id[word]
	y[i] = word2id[target_word[i]] 

X = sequence.pad_sequences(X, maxlen=tbptt_step)
# utils.multi_array_shuffle((X,y))
X_train, X_dev = np.split(X, [int(.9*len(X))])
y_train, y_dev = np.split(y, [int(.9*len(y))])

batch_size = 256
epochs = 40
test = True
units = 50
pretrained_emb_file = '../data/glove.6B.50d.txt'

if test:
	X_train = X_train[:1000]
	X_dev = X_dev[:1000]
	y_train = y_train[:1000]
	y_dev = y_dev[:1000]

embedding_matrix = utils.load_embeddings(pretrained_emb_file, word2id, 50)

model = Sequential()
model.add(Embedding(len(word2id) + 1, 50, input_length=tbptt_step, 
	weights=[embedding_matrix], trainable=True, mask_zero=True))
# model.add(LSTM(256, return_sequences=True))
model.add(GRU(256, return_sequences=True))
model.add(Dropout(0.5))	
# model.add(LSTM(256))
model.add(GRU(256))
model.add(Dropout(0.5))
model.add(Dense(len(word2id) + 1))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) #, perplexity])

model_name = 'test.h5'

if not os.path.isdir('../trained_models'):
	os.makedirs('../trained_models')

early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
model_checkpoint_cb = callbacks.ModelCheckpoint('trained_models/{}'.format(model_name), monitor='val_loss', 
										verbose=0, save_best_only=True, mode='auto')
# perplexity_cb = ComputePerplexityCB()
sentence_gen_cb = SentenceGeneratorCB(word2id, id2word, tbptt_step)

history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_dev,y_dev), epochs=epochs, 
	verbose=1, callbacks=[early_stopping_cb, sentence_gen_cb, model_checkpoint_cb]) # , perplexity_cb])

utils.pickle_dump('../trained_models/{}.p'.format(os.path.basename(model_name)), history.history) 
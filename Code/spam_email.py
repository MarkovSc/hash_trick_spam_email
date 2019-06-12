from tqdm import tqdm
import email
import codecs
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import gc
from textProcess import *

# open file
with open("dataSet/trec07p/full/index") as file:
	fileList= [line.split(' ') for line in file]


from joblib import Parallel, delayed

def parseFile(filename):
	fp = codecs.open("dataSet/trec07p" + filename[1][2:].strip(), "r",encoding='utf-8', errors='ignore')
	email_text = email.message_from_file(fp)
	texts = ""
	if email_text.is_multipart():
		for part in email_text.get_payload():
			if part.get_content_maintype() == 'text':
				texts += part.get_payload()
	else:
		texts += email_text.get_payload()

	texts = re.sub('\n','',texts)
	texts = re.sub('_','',texts)
	texts = re.sub('-',' ',texts)
	texts = re.sub('/',' ',texts)
	texts = re.sub(':',' ',texts)
	texts = re.sub('$','',texts)
	texts = re.sub('=','',texts)
	texts = re.sub('<.*?>', '', texts)
	value = " ".join((BeautifulSoup(texts.lower(), 'html.parser').findAll(text=True)))
	value = preprocess(value)
	fp.close()
	return [int(filename[0] == "spam"), value]

dataList = Parallel(n_jobs= -1, verbose=1)(delayed(parseFile)(i) for i in tqdm(fileList))
print(len(dataList))
df = pd.DataFrame(dataList, columns = ['target' , 'text'])


# text process
from keras.preprocessing import text, sequence
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer

MAX_LEN=100
MAX_FEATURES = 100000
HASH_SIZE = int(1e6)
hash_or_bag=False

if hash_or_bag:
	tt = TweetTokenizer()
	df['sentences'] = df['text'].apply(tt.tokenize)
	cv = CountVectorizer(tokenizer=tt.tokenize)
	vec = cv.fit(list(df['text']))
	bag_of_words = vec.transform(list(df['text']))
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
	words_set = set([x[0] for x in words_freq[:MAX_FEATURES]])

	del vec, bag_of_words, cv, words_freq
	gc.collect()

	def hashFunc(x):
		return int(hash(x) % HASH_SIZE)

	def sequence(x):
		wordList=[hashFunc(word) for word in x if word in words_set]
		return (wordList + [0] * MAX_LEN)[0:MAX_LEN]

	model_all = df['sentences'].apply(sequence).values
	model_data = np.vstack(df['sentences'].apply(sequence).values)
	model_target = df['target'].values
	del df
	gc.collect()

	# embedding for hash table
	EMBEDDING_SIZE=20
	MAX_FEATURES = HASH_SIZE
	embedding_matrix = []
else:
	tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
	tokenizer.fit_on_texts(list(df['text']))
	model_data = tokenizer.texts_to_sequences(df['text'])
	model_data = sequence.pad_sequences(model_data, maxlen=MAX_LEN)
	model_target = df['target'].values
	del df
	gc.collect()

	# embedding
	GLOVE_EMBEDDING_PATH = 'dataSet/Glove/glove.6B.200d.txt'
	EMBEDDING_FILES = [GLOVE_EMBEDDING_PATH]
	EMBEDDING_SIZE=200
	def load_embeddings(path):
		def get_coefs(word, *arr):
			return word, np.asarray(arr, dtype='float32')
		with open(path, encoding="utf-8") as f:
			return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))
	    
	def build_matrix(word_index, path, embed_size = 200):
		embedding_index = load_embeddings(path)
		embedding_matrix = np.zeros((MAX_FEATURES + 1, embed_size))
		unknown_words = {}
		for word, i in word_index.items():
			if(i> MAX_FEATURES):
				break;
			if word in embedding_index:
				embedding_matrix[i] = embedding_index[word]
			elif word.lower() in embedding_index:
				embedding_matrix[i] = embedding_index[word.lower()]
			else:
				if word in unknown_words:
					unknown_words[word] += 1
				else:
					unknown_words[word] = 0;
		'''
		print("unknown word", len(unknown_words))
		uk = [[key,unknown_words[key]] for key in unknown_words]
		uk.sort(key = lambda x: x[1], reverse = True)
		for s in uk[0:100]:
			print(s[0].encode('utf-8'), s[1])
		print("word index", len(word_index))
		'''
		return embedding_matrix

	embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
	del tokenizer
	gc.collect()
	print(embedding_matrix.shape)


from sklearn import metrics
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import time

train_split= int( 0.8 * len(model_target))
train = model_data[:train_split]
test = model_data[train_split:]
target = model_target[:train_split]
target_test = model_target[train_split:]
del model_target, model_data
gc.collect()
print(train.shape)

def train_model(train, test, target, max_len, max_features, embed_size, embedding_matrix, trainable, n_fold =5):
	folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)
	oof = np.zeros((len(train), 1))
	prediction = np.zeros((len(test), 1))
	scores = []
	for fold_n, (train_index, valid_index) in enumerate(folds.split(train, target)):
		print('Fold', fold_n, 'started at', time.ctime())
		X_train, X_valid = train[train_index], train[valid_index]
		y_train, y_valid = target[train_index], target[valid_index]

		model = build_model(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, trainable,
			    lr = 1e-3, lr_d = 0, spatial_dr = 0.0, dense_units=128, conv_size=128, dr=0.05, patience=50)
		model.summary()
		pred_valid = model.predict(X_valid)
		oof[valid_index] = pred_valid
		scores.append(metrics.roc_auc_score(y_valid,pred_valid))

		prediction += model.predict(test, batch_size = 1024, verbose = 1)

	prediction /= n_fold

	# print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
	return oof, prediction, scores

def build_model(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, is_trainable, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.1, patience=3):
	from keras import backend as K
	from keras.models import Sequential, Model
	from keras.layers import BatchNormalization, Input, Embedding, SpatialDropout1D, concatenate, Conv2D, Reshape, Conv1D
	from keras.layers import MaxPool2D, PReLU, AvgPool2D, MaxPooling1D, Bidirectional, LSTM, GRU
	from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
	from keras.layers.core import Flatten, Dense, Dropout, Lambda
	from keras.optimizers import Adam
	from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
	#import Attention
	early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=patience)

	dr = 0.2
	units = 48

	inp = Input(shape = (max_len,))
	if is_trainable:
		emb_desc = Embedding(max_features + 1, embed_size)(inp)
	else:
		emb_desc = Embedding(max_features + 1, embed_size, weights=[embedding_matrix], trainable=False)(inp)

	emb_desc = SpatialDropout1D(.4)(emb_desc)
	#att = Attention(max_len)(emb_desc)
	x1 = SpatialDropout1D(dr)(emb_desc)
	#x = Bidirectional(GRU(units, return_sequences = True))(x1)
	x = GRU(units, return_sequences = True)(x1)
	x = Conv1D(4, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
	#y = Bidirectional(LSTM(units, return_sequences = True))(x1)
	#y = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)

	avg_pool1 = GlobalAveragePooling1D()(x)
	max_pool1 = GlobalMaxPooling1D()(x)
	#avg_pool2 = GlobalAveragePooling1D()(y)
	#max_pool2 = GlobalMaxPooling1D()(y)
	x = concatenate([max_pool1,avg_pool1])
	#x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
	x = BatchNormalization()(x)
	x = Dropout(.3)(x)
	out = Dense(1, activation="sigmoid")(x)

	model = Model(inputs=inp, outputs=out)
	model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
	model.fit(X_train, y_train, batch_size=1000, epochs=5, validation_data=(X_valid, y_valid), 
		verbose=2, callbacks=[early_stop])
	return model



if hash_or_bag:
	oof, prediction, scores = train_model(train, test, target, MAX_LEN, MAX_FEATURES, EMBEDDING_SIZE, embedding_matrix, True)
else:
	oof, prediction, scores = train_model(train, test, target, MAX_LEN, MAX_FEATURES, EMBEDDING_SIZE, embedding_matrix, False)
	
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
print(metrics.roc_auc_score(target_test, prediction))


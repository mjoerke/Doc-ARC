import os
import numpy as np
import pickle
import random
from collections import Counter, OrderedDict

from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)

def read_tagset(filename):
	tags = {}
	with open(filename) as file:
		for line in file:
			cols = line.rstrip().split("\t")
			tags[cols[0]] = int(cols[1])
	return tags


class DocumentAttentionDataset(Dataset):
	"""
	Dataset to be used for document attention training inside of a
	PyTorch DataLoader object
	- tokenizes individual documents
	- computes attention indices 
	"""
	def __init__(self, root_dir, tokenizer, k, self_attention):
		self.root = root_dir
		
		self.cache = os.path.join(self.root, 'cache')
		os.makedirs(self.cache, exist_ok=True)

		self.files = [f for f in os.listdir(root_dir) if 'cache' not in f]
		self.tokenizer = tokenizer
		self.is_cased = tokenizer.init_kwargs['do_lower_case'] 

		self.k = k
		self.self_attention = self_attention

	def __len__(self):
		return len(self.files)

	def load_document(self, file):
		if "brat" in file: # litbank
			doc_id = file.split("_")[0].split('.')[0]
		else:
			doc_id = file.split('.')[0]

		cached_path = os.path.join(self.cache, "{}.tokenized-{}.pkl".format(doc_id, 'cased' if self.is_cased else 'uncased'))

		if os.path.exists(cached_path):
			with open(cached_path, 'rb') as f:
				doc = pickle.load(f)
		else:
			doc_path = os.path.join(self.root, file)
			doc = Document(doc_path)
			doc.tokenize(self.tokenizer)

			with open(cached_path, 'wb') as f:
				pickle.dump(doc, f)

		return doc

	def __getitem__(self, i):
		doc = self.load_document(self.files[i])

		# we want to attend over all possible context, and computational
		# complexity of attention is proportional to context sequence length
		# so we keep context sequences at their original length (pack=False)
		# unless they are longer than max_len (partition=True)
		doc.prepare_sequences(pack=False, partition=True, partition_length=256)
		doc.get_attention(self.tokenizer, self.k, self.self_attention)
		return doc

class SequenceLabelingDataset(Dataset):
	def __init__(self, root_dir, tokenizer):
		self.root = root_dir
		
		self.cache = os.path.join(self.root, 'cache')
		os.makedirs(self.cache, exist_ok=True)

		self.files = [f for f in os.listdir(root_dir) if 'cache' not in f]
		self.tokenizer = tokenizer
		self.is_cased = tokenizer.init_kwargs['do_lower_case'] 

	def __len__(self):
		return len(self.files)

	def load_document(self, file):
		if "brat" in file: # litbank
			doc_id = file.split("_")[0].split('.')[0]
		else:
			doc_id = file.split('.')[0]

		cached_path = os.path.join(self.cache, "{}.tokenized-{}.pkl".format(doc_id, 'cased' if self.is_cased else 'uncased'))

		if os.path.exists(cached_path):
			with open(cached_path, 'rb') as f:
				doc = pickle.load(f)
		else:
			doc_path = os.path.join(self.root, file)
			doc = Document(doc_path)
			doc.tokenize(self.tokenizer)

			with open(cached_path, 'wb') as f:
				pickle.dump(doc, f)

		return doc

	def __getitem__(self, i):
		doc = self.load_document(self.files[i])
		# we don't consider context for labeled sequence labeling
		# so we keep all sequences as is, unless they are too long,
		# in which case they are discarded
		doc.prepare_sequences(pack=False, partition=False)
		return doc


class PretrainingDataset(Dataset):
	def __init__(self, root_dir, tokenizer):
		self.root = root_dir
		
		self.cache = os.path.join(self.root, 'cache')
		os.makedirs(self.cache, exist_ok=True)

		self.files = [f for f in os.listdir(root_dir) if 'cache' not in f]
		self.tokenizer = tokenizer
		self.is_cased = tokenizer.init_kwargs['do_lower_case'] 

	def __len__(self):
		return len(self.files)

	def load_document(self, file):
		if "brat" in file: # litbank
			doc_id = file.split("_")[0].split('.')[0]
		else:
			doc_id = file.split('.')[0]

		cached_path = os.path.join(self.cache, "{}.tokenized-{}.pkl".format(doc_id, 'cased' if self.is_cased else 'uncased'))

		if os.path.exists(cached_path):
			with open(cached_path, 'rb') as f:
				doc = pickle.load(f)
		else:
			doc_path = os.path.join(self.root, file)
			doc = Document(doc_path)
			doc.tokenize(self.tokenizer)

			with open(cached_path, 'wb') as f:
				pickle.dump(doc, f)

		return doc

	def __getitem__(self, i):
		doc = self.load_document(self.files[i])
		# we want to include as much information as possible for MLM
		# pretraining, so we densely pack sequences that are short
		# and partition sequences that are long
		doc.prepare_sequences(pack=True, partition=True)
		doc.generate_masks(masking_prob=0.15, tokenizer=self.tokenizer)
		return doc


class Document():
	"""
	Document data structure for training
	"""

	def __init__(self, path):
		self.path = path
		self.sequences = []
		self.max_len = -1
		
	def tokenize(self, tokenizer, pack_sequences=False):

		self.max_len = tokenizer.max_len

		with open(self.path, encoding='utf8') as f:
			curr_seq = []
			curr_labels = []
			
			for line in f.readlines():
				line = line.rstrip().split("\n")[0]
				
				if line == '':
					curr_seq = [tokenizer.cls_token] + curr_seq + [tokenizer.sep_token]

					if len(curr_labels) > 0:
						curr_labels = [-100] + curr_labels + [-100]
						seq = Sequence(self.path, curr_seq, curr_labels)
					else:
						seq = Sequence(self.path, curr_seq)

					seq.tokenize(tokenizer)
					self.sequences.append(seq)

					curr_seq = []
					curr_labels= []
					
				else:
					word, tag = line.split('\t')
					curr_seq.append(word)

					if tag != 'NULL':
						curr_labels.append(tag)


	def prepare_sequences(self, pack, partition, partition_length=-1):
		"""
		Prepares variable length sequences for various training tasks

		- pack (bool): whether to densely pack sequences up to Document.max_len
			if True: Sequences are packed up until Document.max_len (e.g. for MLM pretraining)
			if False: Sequences are left at their original length (e.g. for supervised seq labeling)

		- partition (bool): whether to split long sequences into subsequences of size partition_length
			if True: unlabeled sequences are split into chunks of size partition_length
					 labeled sequences are never partitioned, only discarded if longer than Document.max_len
			if False: all unlabeled sequences longer than Document.max_len are discarded

		- partition_length: see above, defaults to Document.max_len if set to -1

		"""
		if partition_length == -1:
			partition_length = self.max_len

		if partition:
			sequences = []
			for seq in self.sequences:
				if len(seq) >= partition_length:

					# don't partition labeled sequences, discard if >= max_len
					if seq.is_labeled:
						if len(seq) <= self.max_len:
							sequences.append(seq)
							continue
						else:
							logging.info("Sequence longer than {} encountered in document {}".format(self.max_len, self.path))

					seqs = seq.partition(partition_length)
					sequences += seqs
				else:
					sequences.append(seq)

			self.sequences = sequences


		if pack:
			sequences = []

			running_sequence = self.sequences[0]
			running_length = len(running_sequence) - 2

			for seq in self.sequences[1:]:

				if running_length + len(seq) > self.max_len:
					sequences.append(running_sequence)

					running_sequence = seq
					running_length = len(seq) - 2
				else:
					running_sequence = running_sequence.join(seq)
					running_length = len(running_sequence) - 2

			sequences.append(running_sequence)

			self.sequences = sequences
	 	
		# if both are false, discard all unlabeled sequences > Document.max_len
		if (not pack) and (not partition):
			sequences = []

			for seq in self.sequences:
				if len(seq) > self.max_len and not seq.is_labeled:
					continue
				else:
					sequences.append(seq)


	# generates dict mapping from unique token_id to 
	# list of (sentence idx, token idx) occurence in entire document
	def get_token_positions(self):
		token_to_idx = {}
		
		N = len(self)
		for i in range(N):
			seq = self[i]
			
			T = len(seq)
			for t in range(1, T-1): # exclude [CLS] and [SEP]
				token = seq[t]
				pos = (i, t)
				
				if token in token_to_idx:
					token_to_idx[token].append(pos)
				else:
					token_to_idx[token] = [pos]
		
		return token_to_idx

	# generates dict mapping from words to 
	# list of (sentence idx, token idx) occurence in entire document
	def get_word_positions(self):
		word_to_position = {}
		
		N = len(self)
		for i in range(N):
			seq = self[i]
			
			T = len(seq.words)
			for t in range(1, T-1): # exclude [CLS] and [SEP]
			# for t in range(T):
				# word = seq.words[t].lower()
				word = seq.words[t]
				pos = (i, t)
				
				if word in word_to_position:
					word_to_position[word].append(pos)
				else:
					word_to_position[word] = [pos]
		
		return word_to_position
	
	def get_attention(self, tokenizer, k=5, self_attention=False):
		"""
			returns list of size (# labeled sentences) x (# word) x k
			where attention[i][j] == (seq idx, word idx, distance) for sequences in target doc
									 which contain the same word as word j in 
									 sequence i of source doc 
		"""
				
		word_to_positions = self.get_word_positions()

		N = len(self)
		for i in range(N):
			seq = self[i]

			# don't attend from unlabeled to labeled
			if not seq.is_labeled:
				break

			# sequence level attention - lists of len = # words 
			seq.attn_seq_idxs = []
			seq.attn_word_idxs = []
			seq.attn_dists = []

			T = len(seq.words)
			for t in range(0, T):
				# word = seq.words[t].lower()
				word = seq.words[t]

				if word == tokenizer.cls_token or word == tokenizer.sep_token:
					seq.attn_seq_idxs.append([])
					seq.attn_word_idxs.append([])
					seq.attn_dists.append([])
					continue
				
				occurences = word_to_positions[word]

				if not self_attention:
					occurences = [o for o in occurences if o[0] != i] 
				
				sentence_ids = [o[0] for o in occurences]               
				dist = np.abs(np.array(sentence_ids) - i) # distance from current sentence (i)
														  # to sentences with target token
				if len(dist) <= k:
					idx = np.arange(len(dist))
				else:
					idx = np.argpartition(dist, k)[:k] # get k-smallest elements in arr  

				# word level attention - lists of len <= k
				seq_idxs = []
				word_idxs = []
				dists = []
				for j in idx:
					seq_idx = occurences[j][0]
					seq_idxs.append(seq_idx)

					word_idx = occurences[j][1]
					word_idxs.append(word_idx)

					d = dist[j]
					dists.append(d)

				ordering = np.argsort(dists)

				seq_idxs = [seq_idxs[i] for i in ordering]
				word_idxs = [word_idxs[i] for i in ordering]
				dists = [dists[i] for i in ordering]

				seq.attn_seq_idxs.append(seq_idxs)
				seq.attn_word_idxs.append(word_idxs)
				seq.attn_dists.append(dists)

		self._get_all_attended_sequences()

	def _get_all_attended_sequences(self):
		self.all_attended_seq = []

		for i in range(len(self)):
			seq = self[i]

			if not seq.is_labeled:
				break

			seq_level_attn = seq.attn_seq_idxs

			for word_level_attn in seq_level_attn:
				for seq_idx in word_level_attn:
					self.all_attended_seq.append(seq_idx)

		self.all_attended_seq = np.unique(self.all_attended_seq)
		self.attn_index_map = {}
		for pos, idx in enumerate(self.all_attended_seq):
			self.attn_index_map[idx] = pos  


	def generate_masks(self, masking_prob, tokenizer):
		for seq in self.sequences:
			seq.generate_mask(masking_prob, tokenizer)
				
	# makes the object iterable
	def __getitem__(self, i):
		return self.sequences[i]      
	def __iter__(self):
		self.i = 0
		return self
	def __next__(self):
		if self.i >= self.__len__():
			raise StopIteration
		else:
			value = self.__getitem__(self.i)
			self.i += 1
			return value
	def __len__(self):
		return len(self.sequences)
	
	def num_tokens(self):
		N = 0
		for seq in self.sequences:
			N += len(seq)
		return N

class Sequence():
	"""
	Sequence data structure for training

	arguments:
	- path:     file path to document which contains sequences
	- words:    list of words (strings) in sequence
	- labels:   list of NER labels corresponding to words

	methods:
	- tokenize(tokenizer):  tokenize list of words and generate token indices
	"""

	def __init__(self, path, words, labels=None):
		self.path = path
		self.words = words
		
		if labels:
			self.is_labeled = True
			self.labels = labels
		else:
			self.is_labeled = False
			self.labels = None
		
		self.tokens = None
		self.token_ids = None
		self.word2tokens = []
		self.transform = None
	
	def tokenize(self, tokenizer):
		transform = []
		self.lens = []
		
		self.tokens = []
		
		for w in self.words:
			toks = tokenizer.tokenize(w)
			self.tokens += toks
			self.word2tokens.append(toks)
			self.lens.append(len(toks))

		self.token_ids = tokenizer.convert_tokens_to_ids(self.tokens)
		
		N, T = len(self.words), len(self.tokens)
		self.transform = np.zeros((N, T))
		
		offset = 0
		for i in range(N):
			# self.transform[i][self.word2token[i]] = 1
			for j in range(self.lens[i]):
				self.transform[i][j + offset] = 1. / self.lens[i]
			
			offset += self.lens[i]
		
		return self

	def partition(self, max_len):
		assert not self.is_labeled, "Sequence cannot be labeled"

		# TODO
		if len(self) <= max_len:
			return [self]

		start_word, stop_word = self.words[0], self.words[-1]
		start_tok, stop_tok = self.tokens[0], self.tokens[-1]
		start_tok_id, stop_tok_id = self.token_ids[0], self.token_ids[-1]

		current_tokens = []
		current_words = []
		current_token_ids = []
		current_word2token = []

		word_idx = 1
		token_idx = 1

		partition = []
		while (token_idx < len(self.tokens) - 1):

			if len(current_tokens) + len(self.word2tokens[word_idx]) > max_len - 2:
				words = [start_word] + current_words + [stop_word]
				new_seq = Sequence(self.path, words)
				new_seq.tokens = [start_tok] + current_tokens + [stop_tok]
				new_seq.token_ids = [start_tok_id] + current_token_ids + [stop_tok_id]
				new_seq.word2tokens = [[start_tok]] + current_word2token + [[stop_tok]]
				new_seq.lens = [len(t) for t in new_seq.word2tokens]

				N, T = len(new_seq.words), len(new_seq.tokens)
				new_seq.transform = np.zeros((N, T))

				offset = 0
				for i in range(N):
					for j in range(new_seq.lens[i]):
						new_seq.transform[i][j + offset] = 1. / new_seq.lens[i]
					
					offset += new_seq.lens[i]

				partition.append(new_seq)

				current_tokens = []
				current_words = []
				current_token_ids = []
				current_word2token = []
			else:
				current_words.append(self.words[word_idx])
				current_word2token.append(self.word2tokens[word_idx])

				for _ in range(len(self.word2tokens[word_idx])):
					current_tokens.append(self.tokens[token_idx])
					current_token_ids.append(self.token_ids[token_idx])
					token_idx += 1

				word_idx += 1

		words = [start_word] + current_words + [stop_word]
		new_seq = Sequence(self.path, words)
		new_seq.tokens = [start_tok] + current_tokens + [stop_tok]
		new_seq.token_ids = [start_tok_id] + current_token_ids + [stop_tok_id]
		new_seq.word2tokens = [[start_tok]] + current_word2token + [[stop_tok]]
		new_seq.lens = [len(t) for t in new_seq.word2tokens]

		N, T = len(new_seq.words), len(new_seq.tokens)
		new_seq.transform = np.zeros((N, T))

		offset = 0
		for i in range(N):
			for j in range(new_seq.lens[i]):
				new_seq.transform[i][j + offset] = 1. / new_seq.lens[i]
			
			offset += new_seq.lens[i]
			
		partition.append(new_seq)

		return partition

	def join(self, seq):
		# [CLS,  words_1, SEP] + [CLS,  words_2, SEP] --> [CLS, words_1, words_2, SEP]
		words = self.words[:-1] + seq.words[1:]
		new_seq = Sequence(self.path, words)

		new_seq.tokens = self.tokens[:-1] + seq.tokens[1:]
		new_seq.token_ids = self.token_ids[:-1] + seq.token_ids[1:]
		new_seq.word2tokens = self.word2tokens[:-1] + seq.word2tokens[1:]
		new_seq.lens = [len(t) for t in new_seq.word2tokens]

		N, T = len(new_seq.words), len(new_seq.tokens)
		new_seq.transform = np.zeros((N, T))

		offset = 0
		for i in range(N):
			for j in range(new_seq.lens[i]):
				new_seq.transform[i][j + offset] = 1. / new_seq.lens[i]
			
			offset += new_seq.lens[i]

		return new_seq

	def get_label_ids(self, tagset):
		assert self.is_labeled, "Sequence must be labeled"

		self.label_ids = []

		for i in range(len(self.labels)):
			label = self.labels[i]

			if type(label) is int:
				self.label_ids.append(label)
			elif label in tagset:
				self.label_ids.append(tagset[label])
			else:
				raise ValueError("Unknown label {}".format(label))

		return self.label_ids

	def generate_mask(self, masking_prob, tokenizer):
		# adapted from https://github.com/google-research/bert/blob/master/create_pretraining_data.py

		max_predictions = round((len(self) - 2) * masking_prob)

		# we do whole word masking, so we generate all candidate indices first
		candidate_indices = []
		token_idx = 0

		for i, w in enumerate(self.words):

			if w == tokenizer.sep_token or w == tokenizer.cls_token:
				token_idx += 1
				continue

			toks = self.word2tokens[i]
			token_indices = []

			for _ in toks:
				token_indices.append(token_idx)
				token_idx += 1

			candidate_indices.append(token_indices)

		random.shuffle(candidate_indices)

		num_masked = 0
		original_token_ids = self.token_ids.copy()
		# we don't compute the loss for non-masked tokens
		self.label_ids = [-100] * len(self.token_ids)

		for idx_set in candidate_indices:
			# stop masking once we've reached max predictions
			if num_masked + len(idx_set) > max_predictions:
				break

			for idx in idx_set:
				# 80% of the time, replace with [MASK]
				if random.random() < 0.8:
					masked_token = tokenizer.mask_token_id
				else:
					# 10% of the time, keep original
					if random.random() < 0.5:
						masked_token = original_token_ids[idx]
					# 10% of the time, replace with random token
					else:
						masked_token = random.randint(0, len(tokenizer) - 1)

				self.token_ids[idx] = masked_token
				self.label_ids[idx] = original_token_ids[idx]

				num_masked += 1
		self.is_labeled = True

	
	def __getitem__(self, i):
		return self.token_ids[i]
	def __iter__(self):
		self.i = 0
		return self
	def __next__(self):
		if self.i >= self.__len__():
			raise StopIteration
		else:
			value = self.__getitem__(self.i)
			self.i += 1
			return value
	def __len__(self):
		return len(self.token_ids)
	def __gt__(self, other):
		return len(self) > len(other)
	def __lt__(self, other):
		return len(self) < len(other)
	def __eq__(self, other):
		return len(self) == len(other)
	def __repr__(self):
		return ' '.join(self.words[1:-1])


def get_tf_idf(doc_ids, documents):
	term_document_matrix = {doc_id: Counter() for doc_id in doc_ids}
	document_lengths = {doc_id: 0 for doc_id in doc_ids}

	# compute count(t, d) for all terms in all documents
	for doc_id in doc_ids:
		doc = documents[doc_id]
	
		for sequence in doc.unlabeled_doc:
			for word in sequence.words:
				if word == '[CLS]' or word == '[SEP]':
					continue
				else:
					term_document_matrix[doc_id][word.lower()] += 1
					document_lengths[doc_id] += 1

	max_doc_length = max(document_lengths.values())

	word_counts = Counter()

	for doc_id in doc_ids:
		word_counts += term_document_matrix[doc_id]

	vocab = list(word_counts.keys())
	document_counts = {}

	for word in vocab:
		document_counts[word] = 0
		for doc_id in doc_ids:
			if term_document_matrix[doc_id][word] > 0:
				document_counts[word] += 1

	tf_idf = {doc_id: {} for doc_id in doc_ids}

	N = len(doc_ids)
	for doc_id in doc_ids:
		for word in vocab:
			count_td = term_document_matrix[doc_id][word]
			if count_td > 0:
				count_td *= max_doc_length / document_lengths[doc_id]
				tf = 1 + np.log10(count_td)
				# tf = count_td
				idf = np.log10(N / document_counts[word])
				tf_idf[doc_id][word] = tf * idf
		tf_idf[doc_id]['[cls]'] = 0
		tf_idf[doc_id]['[sep]'] = 0

	return tf_idf
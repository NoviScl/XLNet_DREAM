# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import apex 
import six 

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tokenization_xlnet import XLNetTokenizer
from modeling_xlnet import XLNetConfig, XLNetForSequenceClassification
from optimization import AdamW, WarmupLinearSchedule
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE


import json 

n_class = 3
reverse_order = False 
sa_step = False 

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.text_c = text_c
		self.label = label

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id

class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

# convert to unicode just in case the original data is not 
def convert_to_unicode(text):
	"""Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, bytes):
			return text.decode("utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return text.decode("utf-8", "ignore")
		elif isinstance(text, unicode):
			return text
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")


class dreamProcessor(DataProcessor):
	def __init__(self):
		random.seed(42)
		self.D = [[], [], []]

		for sid in range(3):
			## Note: assuming data folder stored in the same directory 
			with open(["data/train.json", "data/dev.json", "data/test.json"][sid], "r") as f:
				data = json.load(f)
				if sid == 0:
					random.shuffle(data)
				for i in range(len(data)):
					for j in range(len(data[i][1])):
						# still doing simple concat, possible for novelty here!
						d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
						for k in range(len(data[i][1][j]["choice"])):
							d += [data[i][1][j]["choice"][k].lower()]
						d += [data[i][1][j]["answer"].lower()] 
						self.D[sid] += [d]
		
	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[0], "train")

	def get_test_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[2], "test")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[1], "dev")

	def get_labels(self):
		"""See base class."""
		return ["0", "1", "2"]

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, d) in enumerate(data):
			for k in range(3):
				# each d: passage, question, choice * 3, answer 
				if data[i][2+k] == data[i][5]:
					answer = str(k)
					
			label = convert_to_unicode(answer)

			for k in range(3):
				guid = "%s-%s-%s" % (set_type, i, k)
				## passage 
				text_a = convert_to_unicode(data[i][0])
				## choice 
				text_b = convert_to_unicode(data[i][k+2])
				## question 
				text_c = convert_to_unicode(data[i][1])
				examples.append(
						InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))
			
		return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
	"""Loads a data file into a list of `InputBatch`s."""
	# Note: XLNet: A + [SEP] + B + [SEP] + C + [SEP] [CLS]
	# CLS token id for XLNet of 2 
	# pad on left for XLNet 
	cls_tok_id = 2
	pad_tok_id = 4 
	cls_token = tokenizer.cls_token
	sep_token = tokenizer.sep_token

	print("#examples", len(examples))

	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	features = [[]]
	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None

		tokens_c = None
		
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)

		if example.text_c:
			tokens_c = tokenizer.tokenize(example.text_c)

		if tokens_c:
			_truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
			tokens_b = tokens_c + [sep_token] + tokens_b
		elif tokens_b:
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[0:(max_seq_length - 2)]

		tokens = []
		segment_ids = []
		for token in tokens_a:
			tokens.append(token)
			segment_ids.append(0)
		tokens.append(sep_token)
		segment_ids.append(0)

		if tokens_b:
			for token in tokens_b:
				tokens.append(token)
				segment_ids.append(1)
			tokens.append(sep_token)
			segment_ids.append(1)

		tokens.append(cls_token)
		segment_ids.append(cls_tok_id)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length. 
		# pad on left !!! 
		# pad_token = 0
		# pad_segment_id = 4
		padding_length = max_seq_length - len(input_ids)
		input_ids = ([0]*padding_length) + input_ids
		input_mask = ([0]*padding_length) + input_mask
		segment_ids = ([pad_tok_id]*padding_length) + segment_ids

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		label_id = label_map[example.label]
		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("tokens: %s" % " ".join(
					[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label_id))

		features[-1].append(
				InputFeatures(
						input_ids=input_ids,
						input_mask=input_mask,
						segment_ids=segment_ids,
						label_id=label_id))
		## three egs per list 
		if len(features[-1]) == n_class:
			features.append([])

	if len(features[-1]) == 0:
		features = features[:-1]
	print('#features', len(features))
	return features



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
	"""Truncates a sequence tuple in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
		if total_length <= max_length:
			break
		if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
			tokens_a.pop()
		elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
			tokens_b.pop()
		else:
			tokens_c.pop()            


def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs==labels)

def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	
	parser.add_argument("--xlnet_model", default=None, type=str, required=True,
						help="Either one of the two: 'xlnet-large-cased', 'xlnet-base-cased'.")

	parser.add_argument("--output_dir",
						default=None,
						type=str,
						required=True,
						help="The output directory where the model checkpoints will be written.")

	parser.add_argument("--checkpoint_name", default=None, type=str, required=True,
						help="The name of your trained checkpoint.")


	## Other parameters
	parser.add_argument("--max_seq_length",
						default=512,
						type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
							 "Sequences longer than this will be truncated, and sequences shorter \n"
							 "than this will be padded.")
	parser.add_argument("--do_train",
						default=False,
						action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval",
						default=False,
						action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
						default=5e-5,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
						default=3.0,
						type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",
						default=0.1,
						type=float,
						help="Proportion of training to perform linear learning rate warmup for. "
							 "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
						default=False,
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16',
						default=False,
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
						type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	args = parser.parse_args()

	processor = dreamProcessor()
	label_list = processor.get_labels()

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
							args.gradient_accumulation_steps))

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	os.makedirs(args.output_dir, exist_ok=True)

	## only use cased model 
	tokenizer = XLNetTokenizer.from_pretrained(args.xlnet_model, do_lower_case=False) 

	## Load trained model 
	output_model_file = os.path.join(args.output_dir, args.checkpoint_name)
	model_state_dict = torch.load(output_model_file)
	model = XLNetForSequenceClassification.from_pretrained(args.xlnet_model,
        state_dict=model_state_dict,
        num_choices=3)
	logger.info("Trained model: {} loaded.".format(args.checkpoint_name))

	model.to(device)

	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
														  output_device=args.local_rank)
	elif n_gpu > 1:
		model = torch.nn.DataParallel(model)


	if args.do_eval:
		eval_examples = processor.get_test_examples(args.data_dir)
		eval_features = convert_examples_to_features(
			eval_examples, label_list, args.max_seq_length, tokenizer)

		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", len(eval_examples))
		logger.info("  Batch size = %d", args.eval_batch_size)

		input_ids = []
		input_mask = []
		segment_ids = []
		label_id = []
		
		for f in eval_features:
			input_ids.append([])
			input_mask.append([])
			segment_ids.append([])
			for i in range(n_class):
				input_ids[-1].append(f[i].input_ids)
				input_mask[-1].append(f[i].input_mask)
				segment_ids[-1].append(f[i].segment_ids)
			label_id.append([f[0].label_id])                

		all_input_ids = torch.tensor(input_ids, dtype=torch.long)
		all_input_mask = torch.tensor(input_mask, dtype=torch.long)
		all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
		all_label_ids = torch.tensor(label_id, dtype=torch.long)

		eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
		if args.local_rank == -1:
			eval_sampler = SequentialSampler(eval_data)
		else:
			eval_sampler = DistributedSampler(eval_data)
		eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

		model.eval()
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		logits_all = []
		for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			segment_ids = segment_ids.to(device)
			label_ids = label_ids.to(device)

			with torch.no_grad():
				tmp_eval_loss, logits, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, n_class=n_class)

			logits = logits.detach().cpu().numpy()
			label_ids = label_ids.to('cpu').numpy()
			for i in range(len(logits)):
				logits_all += [logits[i]]
			
			tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

			eval_loss += tmp_eval_loss.mean().item()
			eval_accuracy += tmp_eval_accuracy

			nb_eval_examples += input_ids.size(0)
			nb_eval_steps += 1

		eval_loss = eval_loss / nb_eval_steps
		eval_accuracy = eval_accuracy / nb_eval_examples

		if args.do_train:
			result = {'eval_loss': eval_loss,
					  'eval_accuracy': eval_accuracy,
					  'global_step': global_step,
					  'loss': tr_loss/nb_tr_steps}
		else:
			result = {'eval_loss': eval_loss,
					  'eval_accuracy': eval_accuracy}


		output_eval_file = os.path.join(args.output_dir, "eval_results_test.txt")
		with open(output_eval_file, "a+") as writer:
			logger.info("***** Eval results *****")
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))
				writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
	main()

























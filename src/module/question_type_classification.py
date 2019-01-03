from pytorch_pretrained_bert import BertTokenizer # Use BertTokenizer for tokenize
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import os
import sys
import logging
from tqdm import tqdm,trange
import numpy as np

sys.path.append('../model/')

from BiLSTM_SequenceClassifier import BiLSTM_SequenceClassifier

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(2018)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 调用GPU

label_dict = {"解释":0,"是否":1,"方法":2,"命令":3,"原因":4,"时间":5,"地点":6,"查询":7,"状况":8}
max_length = 64
if_union = False
if_bert = False
do_train = True
do_val = True
do_test = False
vocab_size = 21128
embedding_dim = 768
hidden_dim = 128
training_batch_size = 8
validation_batch_size = 8
test_batch_size = 4
num_train_epochs = 3
gradient_accumulation_steps = 1 # Number of updates steps to accumulate before performing a backward/update pass
warmup_proportion = 0.1 # Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10%% of training
learning_rate = 5e-5
model_name = 'BiLSTM_SequenceClassifier'
tokenizer = BertTokenizer.from_pretrained('../../data/bert-base-chinese-vocab.txt')

class InputFeatures(object):
    # A single set of features pf data
    def __init__(self,input_ids,input_mask,segment_ids,label_id,input_length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_length = input_length

def load_data(mode,if_union=False):
    # Load data
    data = []
    filepath = "../../data/type_classification_"
    if if_union:
        filepath += 'union_'
    filepath += mode
    filepath += '.txt'
    file = open(filepath, 'r', encoding='utf-8')
    lines = file.readlines()
    for line in lines:
        label,question = line.strip().split('|')
        question = question.split()
        data.append((label,question))
    return data

def convert_data_to_features(data,max_seq_length,tokenizer,if_bert=True):
    # Loads data into batches
    features = []
    for label,question in data:
        # Since the question here are tokenized, we don't have tokenize it again
        if if_bert == True:
            # Account for [CLS] and [SEP] with "-2"
            if len(question)>max_seq_length-2: # truncate if question is too long
                question = question[:(max_seq_length - 2)]
            # add [CLS] and [SEP]
            tokens = ["[CLS]"] + question + ["[SEP]"]

        else:
            if len(question)>max_seq_length: # truncate if question is too long
                question = question[:max_seq_length]
            tokens = question

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_id = label_dict[label]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to
        input_mask = [1] * len(input_ids)
        input_length = len(input_ids)

        #  Zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # Check whether they are well padded
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id,
                                      input_length=input_length))
    return features

if do_train:
    training_data = load_data('training', if_union)
    training_features = convert_data_to_features(training_data, max_length, tokenizer, if_bert)
    training_record_filepath = "../../record/question_type_classification_training_"
    if if_union:
        training_record_filepath += 'union_'
    training_record_filepath += 'record.txt'
    training_record_file = open(training_record_filepath, 'w')
    training_record_file.write("Training Loss\tTraining Accuracy\n")

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Batch size = %d", training_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in training_features], dtype=torch.long, device=device)
    all_input_mask = torch.tensor([f.input_mask for f in training_features], dtype=torch.long, device=device)
    all_segment_ids = torch.tensor([f.segment_ids for f in training_features], dtype=torch.long, device=device)
    all_label_ids = torch.tensor([f.label_id for f in training_features], dtype=torch.long, device=device)
    all_input_lengths = torch.tensor([f.input_length for f in training_features], dtype=torch.long, device=device)

    training_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lengths)
    training_sampler = RandomSampler(training_dataset)
    training_dataloader = DataLoader(training_dataset, sampler=training_sampler, batch_size=training_batch_size)

    if do_val:
        validation_data = load_data('validation', if_union)
        validation_features = convert_data_to_features(validation_data, max_length, tokenizer, if_bert)

        validation_record_filepath = "../../record/question_type_classification_validation_"
        if if_union:
            validation_record_filepath += 'union_'
        validation_record_filepath += 'record.txt'
        validation_record_file = open(validation_record_filepath, 'w')
        validation_record_file.write("Validation Loss\tValidation Accuracy\n")

        logger.info("***** Validation Set Description *****")
        logger.info("  Num examples = %d", len(validation_features))
        logger.info("  Batch size = %d", validation_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in validation_features], dtype=torch.long, device=device)
        all_input_mask = torch.tensor([f.input_mask for f in validation_features], dtype=torch.long, device=device)
        all_segment_ids = torch.tensor([f.segment_ids for f in validation_features], dtype=torch.long, device=device)
        all_label_ids = torch.tensor([f.label_id for f in validation_features], dtype=torch.long, device=device)
        all_input_lengths = torch.tensor([f.input_length for f in validation_features], dtype=torch.long, device=device)

        validation_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                         all_input_lengths)
        validation_sampler = SequentialSampler(validation_dataset)
        validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=validation_batch_size)

    if model_name == 'BiLSTM_SequenceClassifier':
        model = BiLSTM_SequenceClassifier(vocab_size, label_dict, embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for _ in trange(int(num_train_epochs),desc='Epoch'):
        model.train()
        training_loss = 0
        training_accuracy = 0
        training_steps = 0
        for step,batch in enumerate(tqdm(training_dataloader,desc='Iteration')):
            batch = tuple(t for t in batch)
            input_ids, input_mask, segment_ids, label_ids,input_lengths = batch

            if model_name == 'BiLSTM_SequenceClassifier':
                tmp_training_loss = 0
                tmp_training_accuracy = 0
                tmp_training_steps = 0
                model.zero_grad()
                loss, accuracy = model(input_ids, input_lengths, label_ids, True)
                loss.backward()
                optimizer.step()
                tmp_training_loss += loss.item()
                tmp_training_accuracy += accuracy
                tmp_training_steps += 1

            training_loss += tmp_training_loss
            training_accuracy += tmp_training_accuracy
            training_steps += tmp_training_steps
            print("Training Loss:", training_loss / training_steps)
            print("Training Accuracy:", training_accuracy / training_steps)
            training_record_file.write(str(training_loss / training_steps))
            training_record_file.write('\t')
            training_record_file.write(str(training_accuracy / training_steps))
            training_record_file.write('\n')
        
        if do_val:
            model.eval()
            validation_loss = 0
            validation_accuracy = 0
            validation_steps = 0
            for step, batch in enumerate(tqdm(validation_dataloader, desc='Iteration')):
                batch = tuple(t for t in batch)
                input_ids, input_mask, segment_ids, label_ids, input_lengths = batch

                with torch.no_grad():
                    if model_name == 'BiLSTM_SequenceClassifier':
                        tmp_validation_loss = 0
                        tmp_validation_accuracy = 0
                        tmp_validation_steps = 0
                        loss, accuracy = model(input_ids, input_lengths, label_ids, True)
                        tmp_validation_loss += loss.item()
                        tmp_validation_accuracy += accuracy
                        tmp_validation_steps += 1

                validation_loss += tmp_validation_loss
                validation_accuracy += tmp_validation_accuracy
                validation_steps += 1
                print("Validation Loss:", validation_loss / validation_steps)
                print("Validation Accuracy:", validation_accuracy / validation_steps)
                validation_record_file.write(str(validation_loss / validation_steps))
                validation_record_file.write('\t')
                validation_record_file.write(str(validation_accuracy / validation_steps))
                validation_record_file.write('\n')

    model_path = '../../model/question_type_classification.model'
    torch.save(model.state_dict(), model_path)

if do_test:
    test_data = load_data('test', if_union)
    test_features = convert_data_to_features(test_data, max_length, tokenizer, if_bert)

    test_record_filepath = "../../record/question_type_classification_test_"
    if if_union:
        test_record_filepath += 'union_'
    test_record_filepath += 'record.txt'
    test_record_file = open(test_record_filepath, 'w')
    test_record_file.write("Test Loss\tTest Accuracy\n")

    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_features))
    logger.info("  Batch size = %d", test_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long, device=device)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long, device=device)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long, device=device)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long, device=device)
    all_input_lengths = torch.tensor([f.input_length for f in test_features], dtype=torch.long, device=device)

    test_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lengths)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_steps = 0
    for step, batch in enumerate(tqdm(test_dataloader, desc='Iteration')):
        batch = tuple(t for t in batch)
        input_ids, input_mask, segment_ids, label_ids, input_lengths = batch

        with torch.no_grad():
            if model_name == 'BiLSTM_SequenceClassifier':
                tmp_test_loss = 0
                tmp_test_accuracy = 0
                tmp_test_steps = 0
                loss, accuracy = model(input_ids, input_lengths, label_ids, True)
                tmp_test_loss += loss.item()
                tmp_test_accuracy += accuracy
                tmp_test_steps += 1

        test_loss += tmp_test_loss
        test_accuracy += tmp_test_accuracy
        test_steps += 1
        print("Test Loss:", test_loss / test_steps)
        print("Test Accuracy:", test_accuracy / test_steps)
        test_record_file.write(str(test_loss / test_steps))
        test_record_file.write('\t')
        test_record_file.write(str(test_accuracy / test_steps))
        test_record_file.write('\n')


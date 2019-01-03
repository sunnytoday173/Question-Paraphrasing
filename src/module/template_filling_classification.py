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

from BiLSTM_FillingClassifier import BiLSTM_FillingClassifier

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(2018)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 调用GPU

max_length = 64
if_union = False
if_bert = False
do_train = True
do_val = True
do_test = True
vocab_size = 21128
embedding_dim = 768
hidden_dim = 128
max_position = 5
training_batch_size = 8
validation_batch_size = 8
test_batch_size = 4
num_train_epochs = 3
gradient_accumulation_steps = 1 # Number of updates steps to accumulate before performing a backward/update pass
warmup_proportion = 0.1 # Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10%% of training
learning_rate = 5e-5
model_name = 'BiLSTM_FillingClassifier'
tokenizer = BertTokenizer.from_pretrained('../../data/bert-base-chinese-vocab.txt')

class InputFeatures(object):
    # A single set of features pf data
    def __init__(self,input_ids1, input_mask1, segment_ids1, input_length1, input_ids2, input_mask2, segment_ids2, input_length2, position):
        self.input_ids1 = input_ids1
        self.input_ids2 = input_ids2
        self.input_mask1 = input_mask1
        self.input_mask2 = input_mask2
        self.segment_ids1 = segment_ids1
        self.segment_ids2 = segment_ids2
        self.input_length1 = input_length1
        self.input_length2 = input_length2
        self.position = position

def load_data(mode,if_union=False):
    # Load data
    data = []
    filepath = "../../data/template_filling_"
    if if_union:
        filepath += 'union_'
    filepath += mode
    filepath += '.txt'
    file = open(filepath, 'r', encoding='utf-8')
    lines = file.readlines()
    for line in lines:
        content,template,position = line.strip().split('|')
        content = content.split()
        template = template.split()
        position = int(position)
        data.append((content,template,position))
    return data

def convert_data_to_features(data,max_seq_length,tokenizer,if_bert=True):
    # Loads data into batches
    features = []
    for content,template,position in data:
        # Since the question here are tokenized, we don't have tokenize it again
        if if_bert == True:
            # If we use bert, we only use bert to encode question without further training
            # Account for [CLS] and [SEP] with "-2"
            if len(content)>max_seq_length-2: # truncate if question is too long
                content = content[:(max_seq_length - 2)]
            if len(template)>max_seq_length-2: # truncate if question is too long
                template = template[:(max_seq_length - 2)]
            # add [CLS] and [SEP]
            tokens1 = ["[CLS]"] + content + ["[SEP]"]
            tokens2 = ["[CLS]"] + template + ["[SEP]"]
            input_length1 = len(tokens1)
            input_length2 = len(tokens2)
            segment_ids1 = [0] * len(tokens1)
            segment_ids2 = [0] * len(tokens2)
            input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
            input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to
            input_mask1 = [1] * len(input_ids1)
            input_mask2 = [1] * len(input_ids2)
            #  Zero-pad up to the sequence length
            padding1 = [0] * (max_seq_length - len(input_ids1))
            padding2 = [0] * (max_seq_length - len(input_ids2))
            input_ids1 += padding1
            input_mask1 += padding1
            segment_ids1 += padding1
            input_ids2 += padding2
            input_mask2 += padding2
            segment_ids2 += padding2

            # Check whether they are well padded
            assert len(input_ids1) == max_seq_length
            assert len(input_mask1) == max_seq_length
            assert len(segment_ids1) == max_seq_length
            assert len(input_ids2) == max_seq_length
            assert len(input_mask2) == max_seq_length
            assert len(segment_ids2) == max_seq_length

            features.append(InputFeatures(input_ids1=input_ids1,
                                          input_ids2=input_ids2,
                                          input_mask1=input_mask1,
                                          input_mask2=input_mask2,
                                          segment_ids1=segment_ids1,
                                          segment_ids2=segment_ids2,
                                          input_length1=input_length1,
                                          input_length2=input_length2,
                                          position=position
                                          ))
        else:
            if len(content)> max_seq_length: # truncate if question is too long
                content = content[:max_seq_length]
            if len(template) > max_seq_length:  # truncate if question is too long
                template = template[:max_seq_length]
            tokens1 = content
            tokens2 = template
            segment_ids1 = [0] * len(tokens1)
            segment_ids2 = [0] * len(tokens2)
            input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
            input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
            padding1 = [0] * (max_seq_length - len(input_ids1))
            padding2 = [0] * (max_seq_length - len(input_ids2))
            input_mask1 = [1] * len(input_ids1)
            input_mask2 = [1] * len(input_ids2)
            input_length1 = len(input_ids1)
            input_length2 = len(input_ids2)
            input_ids1 += padding1
            input_mask1 += padding1
            segment_ids1 += padding1
            input_ids2 += padding2
            input_mask2 += padding2
            segment_ids2 += padding2

            features.append(InputFeatures(input_ids1=input_ids1,
                                          input_mask1=input_mask1,
                                          segment_ids1=segment_ids1,
                                          input_length1=input_length1,
                                          input_ids2=input_ids2,
                                          input_mask2=input_mask2,
                                          segment_ids2=segment_ids2,
                                          input_length2=input_length2,
                                          position=position
                                        ))
    return features

if do_train:
    training_data = load_data('training', if_union)
    training_features = convert_data_to_features(training_data, max_length, tokenizer, if_bert)

    training_record_filepath = "../../record/template_filling_classification_training_"
    if if_union:
        training_record_filepath += 'union_'
    training_record_filepath += 'record.txt'
    training_record_file = open(training_record_filepath,'w')
    training_record_file.write("Training Loss\tTraining Accuracy\n")

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Batch size = %d", training_batch_size)

    all_input_ids1 = torch.tensor([f.input_ids1 for f in training_features], dtype=torch.long, device=device)
    all_input_mask1 = torch.tensor([f.input_mask1 for f in training_features], dtype=torch.long, device=device)
    all_segment_ids1 = torch.tensor([f.segment_ids1 for f in training_features], dtype=torch.long, device=device)
    all_input_lengths1 = torch.tensor([f.input_length1 for f in training_features], dtype=torch.long, device=device)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in training_features], dtype=torch.long, device=device)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in training_features], dtype=torch.long, device=device)
    all_segment_ids2 = torch.tensor([f.segment_ids2 for f in training_features], dtype=torch.long, device=device)
    all_input_lengths2 = torch.tensor([f.input_length2 for f in training_features], dtype=torch.long, device=device)
    all_positions = torch.tensor([f.position for f in training_features], dtype=torch.long, device=device)

    training_dataset = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1, all_input_lengths1, all_input_ids2, all_input_mask2, all_segment_ids2, all_input_lengths2, all_positions)
    training_sampler = RandomSampler(training_dataset)
    training_dataloader = DataLoader(training_dataset, sampler=training_sampler, batch_size=training_batch_size)
    
    if do_val:
        validation_data = load_data('validation', if_union)
        validation_features = convert_data_to_features(validation_data, max_length, tokenizer, if_bert)

        validation_record_filepath = "../../record/template_filling_classification_validation_"
        if if_union:
            validation_record_filepath += 'union_'
        validation_record_filepath += 'record.txt'
        validation_record_file = open(validation_record_filepath, 'w')
        validation_record_file.write("Validation Loss\tValidation Accuracy\n")

        logger.info("***** Validation Set Description *****")
        logger.info("  Num examples = %d", len(validation_features))
        logger.info("  Batch size = %d", validation_batch_size)

        all_input_ids1 = torch.tensor([f.input_ids1 for f in validation_features], dtype=torch.long, device=device)
        all_input_mask1 = torch.tensor([f.input_mask1 for f in validation_features], dtype=torch.long, device=device)
        all_segment_ids1 = torch.tensor([f.segment_ids1 for f in validation_features], dtype=torch.long, device=device)
        all_input_lengths1 = torch.tensor([f.input_length1 for f in validation_features], dtype=torch.long, device=device)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in validation_features], dtype=torch.long, device=device)
        all_input_mask2 = torch.tensor([f.input_mask2 for f in validation_features], dtype=torch.long, device=device)
        all_segment_ids2 = torch.tensor([f.segment_ids2 for f in validation_features], dtype=torch.long, device=device)
        all_input_lengths2 = torch.tensor([f.input_length2 for f in validation_features], dtype=torch.long, device=device)
        all_positions = torch.tensor([f.position for f in validation_features], dtype=torch.long, device=device)

        validation_dataset = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1, all_input_lengths1,
                                         all_input_ids2, all_input_mask2, all_segment_ids2, all_input_lengths2,
                                         all_positions)
        validation_sampler = SequentialSampler(validation_dataset)
        validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=validation_batch_size)

    if model_name == 'BiLSTM_FillingClassifier':
        model = BiLSTM_FillingClassifier(vocab_size, max_position, embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for _ in trange(int(num_train_epochs),desc='Epoch'):
        training_loss = 0
        training_accuracy = 0
        training_steps = 0
        if model_name == 'BiLSTM_FillingClassifier':
            model.train()
        for step,batch in enumerate(tqdm(training_dataloader,desc='Iteration')):
            batch = tuple(t for t in batch)
            input_ids1,input_mask1,segment_ids1,input_lengths1,input_ids2,input_mask2,segment_ids2,input_lengths2,positions = batch
            if model_name == 'BiLSTM_FillingClassifier':
                tmp_training_loss = 0
                tmp_training_steps = 0

                if model_name == 'BiLSTM_FillingClassifier':
                    tmp_training_loss = 0
                    tmp_training_accuracy = 0
                    tmp_training_steps = 0
                    model.zero_grad()
                    loss, accuracy = model(input_ids1,input_mask1,input_lengths1,input_ids2,input_mask2,input_lengths2, positions, True)
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
                input_ids1, input_mask1, segment_ids1, input_lengths1, input_ids2, input_mask2, segment_ids2, input_lengths2, positions = batch

                with torch.no_grad():
                    if model_name == 'BiLSTM_FillingClassifier':
                        tmp_validation_loss = 0
                        tmp_validation_accuracy = 0
                        tmp_validation_steps = 0
                        loss, accuracy = model(input_ids1, input_mask1, input_lengths1, input_ids2, input_mask2,
                                               input_lengths2, positions, True)
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

    model_path = '../../model/template_filling_classification.model'
    torch.save(model.state_dict(),model_path)
                    
if do_test:
    test_data = load_data('test', if_union)
    test_features = convert_data_to_features(test_data, max_length, tokenizer, if_bert)

    test_record_filepath = "../../record/template_filling_classification_test_"
    if if_union:
        test_record_filepath += 'union_'
    test_record_filepath += 'record.txt'
    test_record_file = open(test_record_filepath, 'w')
    test_record_file.write("Test Loss\tTest Accuracy\n")

    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_features))
    logger.info("  Batch size = %d", test_batch_size)

    all_input_ids1 = torch.tensor([f.input_ids1 for f in test_features], dtype=torch.long, device=device)
    all_input_mask1 = torch.tensor([f.input_mask1 for f in test_features], dtype=torch.long, device=device)
    all_segment_ids1 = torch.tensor([f.segment_ids1 for f in test_features], dtype=torch.long, device=device)
    all_input_lengths1 = torch.tensor([f.input_length1 for f in test_features], dtype=torch.long, device=device)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in test_features], dtype=torch.long, device=device)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in test_features], dtype=torch.long, device=device)
    all_segment_ids2 = torch.tensor([f.segment_ids2 for f in test_features], dtype=torch.long, device=device)
    all_input_lengths2 = torch.tensor([f.input_length2 for f in test_features], dtype=torch.long, device=device)
    all_positions = torch.tensor([f.position for f in test_features], dtype=torch.long, device=device)

    test_dataset = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1, all_input_lengths1,
                                     all_input_ids2, all_input_mask2, all_segment_ids2, all_input_lengths2,
                                     all_positions)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_steps = 0
    for step, batch in enumerate(tqdm(test_dataloader, desc='Iteration')):
        batch = tuple(t for t in batch)
        input_ids1, input_mask1, segment_ids1, input_lengths1, input_ids2, input_mask2, segment_ids2, input_lengths2, positions = batch

        with torch.no_grad():
            if model_name == 'BiLSTM_FillingClassifier':
                tmp_test_loss = 0
                tmp_test_accuracy = 0
                tmp_test_steps = 0
                loss, accuracy = model(input_ids1, input_mask1, input_lengths1, input_ids2, input_mask2, input_lengths2,
                                       positions, True)
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

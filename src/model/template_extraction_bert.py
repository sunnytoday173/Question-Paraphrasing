from pytorch_pretrained_bert import BertTokenizer # Use BertTokenizer for tokenize
from pytorch_pretrained_bert import BertModel,BertForTokenClassification
from pytorch_pretrained_bert import BertAdam
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import numpy as np
from tqdm import tqdm,trange
import os
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(2018)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 调用GPU

label_dict = {"B":0,"I":1,"O":2}
max_length = 64
if_union = False
if_bert = True
do_train = True
do_eval = True
training_batch_size = 8
validation_batch_size = 4
num_train_epochs = 3
gradient_accumulation_steps = 1 # Number of updates steps to accumulate before performing a backward/update pass
warmup_proportion = 0.1 # Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10%% of training
learning_rate = 5e-5
tokenizer = BertTokenizer.from_pretrained('../../data/bert-base-chinese-vocab.txt')

class InputFeatures(object):
    # A single set of features pf data
    def __init__(self,input_ids,input_mask,segment_ids,label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def load_data(mode,if_union=False):
    # Load data
    data = []
    filepath = "../../data/template_extraction_"
    if if_union:
        filepath += 'union_'
    filepath += mode
    filepath += '.txt'
    file = open(filepath, 'r', encoding='utf-8')
    lines = file.readlines()
    for line in lines:
        question,label = line.strip().split('|')
        question = question.split()
        label = label.split()
        data.append((question,label))
    return data

def convert_data_to_features(data,max_seq_length,tokenizer,if_bert=True):
    # Loads data into batches
    features = []
    for question,label in data:
        # Since the question here are tokenized, we don't have tokenize it again
        if if_bert == True:
            # Account for [CLS] and [SEP] with "-2"
            if len(question)>max_seq_length-2: # truncate if question is too long
                question = question[:(max_seq_length - 2)]
                label = label[:(max_seq_length - 2)]
            # add [CLS] and [SEP]
            tokens = ["[CLS]"] + question + ["[SEP]"]
            label = ["O"] + label + ["O"]
        else:
            if len(question)>max_seq_length: # truncate if question is too long
                question = question[:max_seq_length]
                label = label[:max_seq_length]
            tokens = question

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_dict[item] for item in label]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to
        input_mask = [1] * len(input_ids)

        #  Zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_ids += padding

        # Check whether they are well padded
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_ids=label_ids))
    return features

def warmup_linear(x,warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def accuracy(out, labels):
    # Accuracy metric
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

layer_indexes = [-1,-2,-3,-4]


global_step = 0

if do_train:
    training_data = load_data('training',if_union)
    training_features = convert_data_to_features(training_data,max_length,tokenizer,if_bert)
    all_input_ids = torch.tensor([f.input_ids for f in training_features],dtype=torch.long,device=device)
    all_input_mask = torch.tensor([f.input_mask for f in training_features],dtype=torch.long,device=device)
    all_segment_ids = torch.tensor([f.segment_ids for f in training_features],dtype=torch.long,device=device)
    all_label_ids = torch.tensor([f.label_ids for f in training_features],dtype=torch.long,device=device)

    training_dataset = TensorDataset(all_input_ids,all_input_mask,all_segment_ids,all_label_ids)
    training_sampler = RandomSampler(training_dataset)
    training_dataloader = DataLoader(training_dataset,sampler=training_sampler,batch_size=training_batch_size)

    num_train_steps = int(len(training_features)/training_batch_size/gradient_accumulation_steps*num_train_epochs)
    t_total = num_train_steps

    if if_bert:
        model = BertForTokenClassification.from_pretrained('../../model/bert-base-chinese.tar.gz',num_labels=3).to(device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=t_total)

    model.train() #Turn to training mode
    for _ in trange(int(num_train_epochs),desc='Epoch'):
        training_loss = 0
        training_steps = 0
        for step,batch in enumerate(tqdm(training_dataloader,desc='Iteration')):
            batch = tuple(t for t in batch)
            input_ids,input_mask,segment_ids,label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            # if we have more than one gpu,we need to average the loss

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            training_loss += loss.item()
            training_steps += 1
            if (step+1) % gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = learning_rate * warmup_linear(global_step / t_total, warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join('../../model/', "template_extraction_model_bert.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model = BertForTokenClassification.from_pretrained('../../model/bert-base-chinese.tar.gz', num_labels=3,state_dict=model_state_dict)
    model.to(device)

if do_eval:
    validation_data = load_data('training', if_union)
    validation_features = convert_data_to_features(validation_data, max_length, tokenizer, if_bert)
    all_input_ids = torch.tensor([f.input_ids for f in validation_features], dtype=torch.long, device=device)
    all_input_mask = torch.tensor([f.input_mask for f in validation_features], dtype=torch.long, device=device)
    all_segment_ids = torch.tensor([f.segment_ids for f in validation_features], dtype=torch.long, device=device)
    all_label_ids = torch.tensor([f.label_ids for f in validation_features], dtype=torch.long, device=device)
    validation_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    validation_sampler = SequentialSampler(validation_dataset)
    validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=validation_batch_size)

    output_model_file = os.path.join('../../model/', "template_extraction_model_bert.bin")
    model_state_dict = torch.load(output_model_file)
    model = BertForTokenClassification.from_pretrained('../../model/bert-base-chinese.tar.gz',
                                                       state_dict=model_state_dict)
    model.to(device)
    model.eval() # turn to eval mode
    validation_loss,validation_accuracy = 0, 0
    nb_validation_steps, nb_validation_examples = 0, 0

    for input_ids,input_mask,segment_ids,label_ids in validation_dataloader:
        with torch.no_grad():
            tmp_validation_loss = model(input_ids,segment_ids,input_mask,label_ids)
            logits = model(input_ids, segment_ids, input_mask)# This line should be the predicted result

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        #tmp_eval_accuracy = accuracy(logits, label_ids)
        tmp_validation_accuracy = 0

        validation_loss += tmp_validation_loss.mean().item()
        validation_accuracy += tmp_validation_accuracy

        nb_validation_examples += input_ids.size(0)
        nb_validation_steps += 1

    eval_loss = validation_loss / nb_validation_steps
    eval_accuracy = validation_accuracy / nb_validation_examples

    result = {'validation_loss': validation_loss,
              'validation_accuracy': validation_accuracy}

    output_eval_file = os.path.join('../../result/', "template_extraction_bert_eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))
from pytorch_pretrained_bert import BertTokenizer # Use BertTokenizer for tokenize
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import os
import sys
import random
import logging
from tqdm import tqdm,trange
import numpy as np

sys.path.append('../model/')

from Seq2Seq import EncoderRNN,AttnDecoderRNN

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

random.seed(2018)
torch.manual_seed(2018)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 调用GPU

max_length = 64
if_union = False
if_bert = False
do_train = True
do_val = True
do_test = False
vocab_size = 21128
embedding_dim = 768
hidden_dim = 256
training_batch_size = 8
validation_batch_size = 8
test_batch_size = 4
num_train_epochs = 3
teacher_forcing_ratio = 0.5
gradient_accumulation_steps = 1 # Number of updates steps to accumulate before performing a backward/update pass
warmup_proportion = 0.1 # Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10%% of training
learning_rate = 5e-5
model_name = 'Seq2Seq'
tokenizer = BertTokenizer.from_pretrained('../../data/bert-base-chinese-vocab.txt')

class InputFeatures(object):
    # A single set of features pf data
    def __init__(self,input_ids1, input_mask1, segment_ids1, input_length1, input_ids2=None, input_mask2=None, segment_ids2=None, input_length2=None):
        self.input_ids1 = input_ids1
        self.input_ids2 = input_ids2
        self.input_mask1 = input_mask1
        self.input_mask2 = input_mask2
        self.segment_ids1 = segment_ids1
        self.segment_ids2 = segment_ids2
        self.input_length1 = input_length1
        self.input_length2 = input_length2

def load_data(mode,if_union=False):
    # Load data
    data = []
    filepath = "../../data/template_transforming_"
    if if_union:
        filepath += 'union_'
    filepath += mode
    filepath += '.txt'
    file = open(filepath, 'r', encoding='utf-8')
    lines = file.readlines()
    for line in lines:
        question1,question2 = line.strip().split('|')
        question1 = question1.split()
        question2 = question2.split()
        data.append((question1,question2))
    return data

def convert_data_to_features(data,max_seq_length,tokenizer,if_bert=True):
    # Loads data into batches
    features = []
    for question1,question2 in data:
        # Since the question here are tokenized, we don't have tokenize it again
        if if_bert == True:
            # If we use bert, we only use bert to encode question without further training
            # Account for [CLS] and [SEP] with "-2"
            if len(question1)>max_seq_length-2: # truncate if question is too long
                question1 = question1[:(max_seq_length - 2)]
            # add [CLS] and [SEP]
            tokens1 = ["[CLS]"] + question1 + ["[SEP]"]
            input_lengths = len(tokens1)
            segment_ids = [0] * len(tokens1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens1)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to
            input_mask = [1] * len(input_ids)
            #  Zero-pad up to the sequence length
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            # Check whether they are well padded
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(InputFeatures(input_ids1=input_ids,
                                          input_mask1=input_mask,
                                          segment_ids1=segment_ids,
                                          input_length1=input_lengths
                                          ))

        else:
            if len(question1)> max_seq_length: # truncate if question is too long
                question1 = question1[:max_seq_length]
            if len(question2) > max_seq_length:  # truncate if question is too long
                question2 = question2[:max_seq_length]
            tokens1 = question1 + ["[SEP]"] # The CLS and SEP here represents SOS and EOS token
            tokens2 = question2 + ["[SEP]"]
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
                                        ))
    return features

def build_index(encoder,if_union=False):
    import pickle
    filepath = "../../data/pickle/template_set"
    if if_union:
        filepath += '_union'
    filepath += '.pickle'
    file = open(filepath, 'rb')
    template_set = pickle.load(file)
    index = dict()
    for template in template_set:
        tokens = template.split(' ')
        template_ids = tokenizer.convert_tokens_to_ids(tokens)
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(len(template_ids)):
            encoder_output, encoder_hidden = encoder(template_ids[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        index[template] = encoder_hidden.detach().cpu().numpy().flatten()
    print("Index Built")
    return index

def retrieval(index,template_representation,slot_num=None,k=4):
    max_similarity = []
    embedding = template_representation
    embedding_norm = np.linalg.norm(embedding)
    for target_sentence, target_embedding in index.items():
        target_embedding_norm = np.linalg.norm(target_embedding)

        similarity = np.dot(embedding, target_embedding)
        similarity /= (embedding_norm * target_embedding_norm)
        if slot_num is not None:
            target_slot_num = target_sentence.count('#')
            if target_slot_num == slot_num:
                if len(max_similarity) < k:
                    max_similarity.append((target_sentence, target_embedding, similarity))
                    max_similarity = sorted(max_similarity, key=lambda x: x[2], reverse=True)
                elif similarity > max_similarity[-1][2] and target_slot_num == slot_num:
                    relevant_sentence = target_sentence
                    max_similarity[-1] = (relevant_sentence, target_embedding, similarity)
                    max_similarity = sorted(max_similarity, key=lambda x: x[2], reverse=True)
        else:
            if len(max_similarity) < k:
                max_similarity.append((target_sentence, target_embedding, similarity))
                max_similarity = sorted(max_similarity, key=lambda x: x[2], reverse=True)
            elif similarity > max_similarity[-1][2]:
                relevant_sentence = target_sentence
                max_similarity[-1] = (relevant_sentence, target_embedding, similarity)
                max_similarity = sorted(max_similarity, key=lambda x: x[2], reverse=True)
    return max_similarity




if do_train:
    training_data = load_data('training', if_union)
    training_features = convert_data_to_features(training_data, max_length, tokenizer, if_bert)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Batch size = %d", training_batch_size)

    if if_bert==True:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in training_features], dtype=torch.long, device=device)
        all_input_mask = torch.tensor([f.input_mask for f in training_features], dtype=torch.long, device=device)
        all_segment_ids = torch.tensor([f.segment_ids for f in training_features], dtype=torch.long, device=device)
        training_dataset = TensorDataset(all_input_ids1, all_input_mask, all_segment_ids)
    else:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in training_features], dtype=torch.long, device=device)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in training_features], dtype=torch.long, device=device)
        all_input_length1 = torch.tensor([f.input_length1 for f in training_features], dtype=torch.long, device=device)
        all_input_length2 = torch.tensor([f.input_length2 for f in training_features], dtype=torch.long, device=device)
        training_dataset = TensorDataset(all_input_ids1, all_input_ids2, all_input_length1, all_input_length2)


    training_sampler = RandomSampler(training_dataset)
    training_dataloader = DataLoader(training_dataset, sampler=training_sampler, batch_size=training_batch_size)

    if do_val:
        validation_data = load_data('validation', if_union)
        validation_features = convert_data_to_features(validation_data, max_length, tokenizer, if_bert)

        logger.info("***** Validation Set Description *****")
        logger.info("  Num examples = %d", len(validation_features))
        logger.info("  Batch size = %d", validation_batch_size)

        if if_bert == True:
            all_input_ids1 = torch.tensor([f.input_ids1 for f in validation_features], dtype=torch.long, device=device)
            all_input_mask = torch.tensor([f.input_mask for f in validation_features], dtype=torch.long, device=device)
            all_segment_ids = torch.tensor([f.segment_ids for f in validation_features], dtype=torch.long, device=device)
            validation_dataset = TensorDataset(all_input_ids1, all_input_mask, all_segment_ids)
        else:
            all_input_ids1 = torch.tensor([f.input_ids1 for f in validation_features], dtype=torch.long, device=device)
            all_input_ids2 = torch.tensor([f.input_ids2 for f in validation_features], dtype=torch.long, device=device)
            all_input_length1 = torch.tensor([f.input_length1 for f in validation_features], dtype=torch.long,
                                             device=device)
            all_input_length2 = torch.tensor([f.input_length2 for f in validation_features], dtype=torch.long,
                                             device=device)
            validation_dataset = TensorDataset(all_input_ids1, all_input_ids2, all_input_length1, all_input_length2)

        validation_sampler = SequentialSampler(validation_dataset)
        validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=validation_batch_size)

    if model_name == 'Seq2Seq':
        encoder = EncoderRNN(vocab_size, embedding_dim, hidden_dim).to(device)
        decoder = AttnDecoderRNN(hidden_dim, embedding_dim, vocab_size, dropout_p=0.1).to(device)
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

    for _ in trange(int(num_train_epochs),desc='Epoch'):
        training_loss = 0
        training_accuracy = 0
        training_steps = 0
        if model_name == 'Seq2Seq':
            encoder.train()
            decoder.train()
        for step,batch in enumerate(tqdm(training_dataloader,desc='Iteration')):
            batch = tuple(t for t in batch)
            if if_bert == True:
                input_ids1, input_mask, segment_ids = batch
            else:
                input_ids1, input_ids2, input_lengths1, input_lengths2 = batch
            if model_name == 'Seq2Seq':
                tmp_training_loss = 0
                tmp_training_steps = 0
                for input_id1, input_id2, input_length1,input_length2 in zip(input_ids1, input_ids2, input_lengths1, input_lengths2):
                    encoder.zero_grad()
                    decoder.zero_grad()
                    loss = 0.
                    input_id1 = input_id1[:input_length1]
                    input_id2 = input_id2[:input_length2].view(-1,1)

                    encoder_hidden = encoder.initHidden()
                    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

                    # Encoder part
                    for ei in range(input_length1):
                        encoder_output, encoder_hidden = encoder(
                            input_id1[ei], encoder_hidden)
                        encoder_outputs[ei] = encoder_output[0, 0]

                    # Decoder part
                    SOS_token = tokenizer.convert_tokens_to_ids(["[CLS]"]) # Here "CLS" represents SOS_token
                    decoder_input = torch.tensor([SOS_token], device=device)
                    decoder_hidden = encoder_hidden
                    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False  # 是否采用teacher_forcing

                    if use_teacher_forcing:
                        # Teacher forcing: 将真实值作为下一个位置的输入
                        for di in range(input_length2):
                            decoder_output, decoder_hidden, decoder_attention = decoder(
                                decoder_input, decoder_hidden, encoder_outputs)
                            loss += criterion(decoder_output, input_id2[di])
                            decoder_input = input_id2[di]  # Teacher forcing
                    else:
                        # Without teacher forcing: 将预测值作为下一个位置的输入
                        for di in range(input_length2):
                            decoder_output, decoder_hidden, decoder_attention = decoder(
                                decoder_input, decoder_hidden, encoder_outputs)
                            topv, topi = decoder_output.topk(1)
                            decoder_input = topi.squeeze().detach()  # detach from history as input
                            loss += criterion(decoder_output, input_id2[di])
                            if decoder_input.item() == "[SEP]":
                                break
                    loss = loss / input_length2.item()  # Normalization may be not necessary
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    tmp_training_loss += loss.item()
                    tmp_training_steps += 1
                tmp_training_loss /= tmp_training_steps
            training_loss += tmp_training_loss
            training_steps += tmp_training_steps
            print("Training Loss:", training_loss / training_steps)

    encoder_path = '../../model/template_transforming_encoder.model'
    decoder_path = '../../model/template_transforming_decoder.model'
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

index = build_index(encoder)

if do_test:
    test_data = load_data('test', if_union)
    test_features = convert_data_to_features(test_data, max_length, tokenizer, if_bert)

    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_features))
    logger.info("  Batch size = %d", test_batch_size)

    if if_bert == True:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in test_features], dtype=torch.long, device=device)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long, device=device)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long, device=device)
        test_dataset = TensorDataset(all_input_ids1, all_input_mask, all_segment_ids)
    else:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in test_features], dtype=torch.long, device=device)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in test_features], dtype=torch.long, device=device)
        all_input_length1 = torch.tensor([f.input_length1 for f in test_features], dtype=torch.long, device=device)
        all_input_length2 = torch.tensor([f.input_length2 for f in test_features], dtype=torch.long, device=device)
        test_dataset = TensorDataset(all_input_ids1, all_input_ids2, all_input_length1, all_input_length2)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

    if model_name == 'Seq2Seq':
        encoder.eval()
        decoder.eval()
    for step, batch in enumerate(tqdm(test_dataloader, desc='Iteration')):
        batch = tuple(t for t in batch)
        if if_bert == True:
            input_ids1, input_mask, segment_ids = batch
        else:
            input_ids1, input_ids2, input_lengths1, input_lengths2 = batch
        if model_name == 'Seq2Seq':
            tmp_training_loss = 0
            tmp_training_steps = 0
            for input_id1, input_id2, input_length1, input_length2 in zip(input_ids1, input_ids2, input_lengths1,
                                                                          input_lengths2):
                encoder.zero_grad()
                decoder.zero_grad()
                loss = 0.
                input_id1 = input_id1[:input_length1]
                input_id2 = input_id2[:input_length2].view(-1, 1)

                encoder_hidden = encoder.initHidden()
                encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

                # Encoder part
                for ei in range(input_length1):
                    encoder_output, encoder_hidden = encoder(
                        input_id1[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]

                sentence_representation = encoder_hidden.cpu().detach()
                max_similarity = retrieval(index,sentence_representation)
                print(max_similarity[0][0], max_similarity[0][2])














# encoding:utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 调用GPU

class BiLSTM_TokenClassifier(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_TokenClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True,batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

    def forward(self, input_ids,input_lengths,input_mask=None,label_ids=None,show_accuracy=False):
        # Convert word indexes to embeddings
        embedded = self.word_embeds(input_ids)
        # Sort idx for packing
        #seq_length = input_ids.size()[1]
        #print(seq_length)
        sorted, x_sort_idx = torch.sort(-input_lengths)
        unsorted, x_unsort_idx = torch.sort(x_sort_idx)
        input_lengths = input_lengths[x_sort_idx]
        embedded = embedded[x_sort_idx]
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,batch_first=True)
        # Forward pass through LSTM
        outputs, _ = self.lstm(packed)
        # Unpack padding
        outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True) # batch_size * max_length(in the batch) * hidden_dim
        # Recover it to the original order
        outputs = outputs[x_unsort_idx]
        #print('Output Size:',outputs.size())
        tags = self.hidden2tag(outputs) # batch_size * max_length(in the batch) * tagset_size
        #print('Tag Size:',tags.size())
        if label_ids is not None:
            new_seq_length = tags.size()[1]
            label_ids = label_ids[:,:new_seq_length]
            #print('Label_ids Size:', label_ids.size())
            criterion = nn.CrossEntropyLoss()
            loss = criterion(tags.view(-1,self.tagset_size),label_ids.flatten()) # view(-1) will produce error here probably due to pytorch 0.4.1
            if show_accuracy == True:
                input_mask = input_mask[:, :new_seq_length]
                pred_label = torch.argmax(tags,dim=2)
                total = torch.sum(input_mask).item()
                correct = torch.sum(torch.mul((pred_label == label_ids).type_as(input_mask),input_mask)).item()
                accuracy = correct*1.0/total
                return loss,accuracy
            return loss
        return tags

# encoding:utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU device

class BiLSTM_SequenceClassifier(nn.Module):

    def __init__(self, vocab_size, label_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_SequenceClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.labelset_size = len(label_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True,batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, self.labelset_size)

    def forward(self, input_ids,input_lengths,label_ids=None,show_accuracy=False):
        # Convert word indexes to embeddings
        embedded = self.word_embeds(input_ids)
        # Sort idx for packing
        sorted, x_sort_idx = torch.sort(-input_lengths)
        unsorted, x_unsort_idx = torch.sort(x_sort_idx)
        input_lengths = input_lengths[x_sort_idx]
        embedded = embedded[x_sort_idx]
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        # Forward pass through LSTM
        outputs, (hidden,_) = self.lstm(packed)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)  # batch_size * max_length(in the batch) * hidden_dim
        # Generate features
        hidden = hidden.transpose(0, 1)
        # Recover it to the original order
        hidden = hidden[x_unsort_idx]
        # Stack two directions together
        hidden = hidden.contiguous().view(-1, self.hidden_dim)
        #print(features.size())
        label = self.hidden2label(hidden)  # batch_size * tagset_size
        if label_ids is not None:
            label_ids = label_ids
            #print('Label_ids Size:', label_ids.size())
            criterion = nn.CrossEntropyLoss()
            loss = criterion(label.view(-1,self.labelset_size),label_ids.flatten()) # view(-1) will produce error here probably due to pytorch 0.4.1
            if show_accuracy == True:
                pred_label = torch.argmax(label,dim=1)
                total = pred_label.size()[0]
                correct = torch.sum((pred_label == label_ids)).item()
                accuracy = correct*1.0/total
                return loss,accuracy
            return loss
        return label
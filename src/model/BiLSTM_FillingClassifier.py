# encoding:utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU device

class BiLSTM_FillingClassifier(nn.Module):
    # 定义模型
    def __init__(self, vocab_size, max_position, embedding_dim, hidden_dim):
        super(BiLSTM_FillingClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_position = max_position

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.template_encoder = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,bidirectional=True,batch_first=True)  # Template Encoder
        self.content_encoder = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,bidirectional=True,batch_first=True)  # Content Encoder

        # 由于是分类任务，因此要将隐状态映射到标注空间，可能的问题 类别数不固定
        self.hidden2position = nn.Linear(2 * hidden_dim, self.max_position)


    def forward(self,input_ids1,input_mask1,input_lengths1,input_ids2,input_mask2,input_lengths2, positions=None, show_accuracy=True):
        content_embeds = self.word_embeds(input_ids1)
        template_embeds = self.word_embeds(input_ids2)
        # Get batch_size
        batch_size = content_embeds.size()[0]
        # Sort idx for packing
        content_sorted, content_sort_idx = torch.sort(-input_lengths1)
        content_unsorted, content_unsort_idx = torch.sort(content_sort_idx)
        input_lengths1 = input_lengths1[content_sort_idx]
        template_sorted, template_sort_idx = torch.sort(-input_lengths2)
        template_unsorted, template_unsort_idx = torch.sort(template_sort_idx)
        input_lengths2 = input_lengths2[template_sort_idx]
        content_embeds = content_embeds[content_sort_idx]
        template_embeds = template_embeds[template_sort_idx]
        #print(content_embeds.size())
        #print(template_embeds.size())
        # Pack padded batch of sequences for RNN module
        content_packed = torch.nn.utils.rnn.pack_padded_sequence(content_embeds, input_lengths1, batch_first=True)
        template_packed = torch.nn.utils.rnn.pack_padded_sequence(template_embeds, input_lengths2, batch_first=True)
        # Forward pass through LSTM
        content_lstm_out, (content_hidden, _) = self.content_encoder(content_packed)
        template_lstm_out, (template_hidden,_) = self.template_encoder(template_packed)
        # Transpose to make it batch first
        content_hidden = content_hidden.transpose(0,1)
        template_hidden = template_hidden.transpose(0,1)
        # Unsort idx
        content_hidden = content_hidden[content_unsort_idx] #
        template_hidden = template_hidden[template_unsort_idx]
        # stack two directions together
        content_hidden = content_hidden.contiguous().view(-1,self.hidden_dim)
        template_hidden = template_hidden.contiguous().view(-1,self.hidden_dim)

        #print(content_hidden.size())
        #print(template_hidden.size())
        hidden_concated = torch.cat((template_hidden,content_hidden),1)
        #print(hidden_concated.size())
        pred_positions = self.hidden2position(hidden_concated)
        if positions is not None:
            #print('Label_ids Size:', label_ids.size())
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred_positions,positions) # view(-1) will produce error here probably due to pytorch 0.4.1
            if show_accuracy == True:
                pred_label = torch.argmax(pred_positions,dim=1)
                total = pred_label.size()[0]
                correct = torch.sum((pred_label == positions)).item()
                accuracy = correct*1.0/total
                return loss,accuracy
            return loss
        return pred_positions
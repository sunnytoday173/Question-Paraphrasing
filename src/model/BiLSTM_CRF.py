# encoding:utf-8
# 该BiLSTM-CRF模型主要参考Pytorch Tutorial中的BiLSTM-CRF部分 原作者:Robert Guthrie
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

START_TAG = "<START>"
STOP_TAG = "<STOP>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 调用GPU

def argmax(vec):
    # 返回最大值位置,torch.max 返回的第一个值为最大值，第二个为位置
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# 前向传播，计算log sum exp，这部分需要后续再参考下原文
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    # 定义BiLSTM_CRF模型
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)  # 这里是双向LSTM 因此hidden dim 与 bidirectional有所调整

        # 由于是序列标注任务，因此要将隐状态映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移概率矩阵 [i,j] 表示从i状态转移到j状态的分数(不称为概率因为没做归一化)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 由于不可能从任何状态转移到起始状态(不是说词的开始而是句子开始）也不可能从结束状态开始转移，因此设置特殊值
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2,device=device),
                torch.randn(2, 1, self.hidden_dim // 2,device=device))

    def _forward_alg(self, feats):
        # 前向算法，计算配分函数
        # 初始化
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        # 起始标志获得全局得分
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 初始状态化forward_var，随着step变化，用于梯度回传
        forward_var = init_alphas

        # 按句子中位置进行迭代
        for feat in feats: # feat应该是指feature
            alphas_t = []  # 该时间步的前向张量
            for next_tag in range(self.tagset_size):
                # 传播emission score
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # transition score的第i个位置是从i转移到next_tag的score (ith entry 不是很明白)
                trans_score = self.transitions[next_tag].view(1, -1)
                # 在我们做log-sum-exp之前，next_tag_var的第i个位置是 i -> next_tag 的边的值（应该就是说三个求和）
                next_tag_var = forward_var + trans_score + emit_score
                # 这个tag的前向变量就是所有分数的log_sum_exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 给句子打分
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long,device=device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 用Viterbi算法进行解码，预测序列及对应标签
        backpointers = []

        # 初始化Viterbi中的变量（似乎是取了log但这里好像没体现)
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # 第i步的forward_var存储了i-1步的Viterbi变量
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 存储这个时间步的backpointers
            viterbivars_t = []  # 存储这个时间步的Viterbi变量

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] 存储了前一个时间步Viterbi变量的值+从第i个tag转移到next_tag的值
                # 这里不考虑emission socres是因为最后有个取argmax的过程，常量不起作用（说是在后面加)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 在此处加入emission score 然后给forward_var赋值(即三个值的和)
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 转移到STOP_TAG的值赋给Transition
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 根据backpointers选取最佳路径（由后往前，因此最后需要倒序）
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 把start tag拿出来
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 可用性测试
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, show_accuracy = False):
        # 负对数似然，事实上计算过程已经在上面的过程中体现了
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        if show_accuracy:
            score, tag_seq = self._viterbi_decode(feats)
            tag_seq = torch.tensor(tag_seq,device=device)
            correct = torch.sum(tag_seq==tags).item()
            accuracy = correct*1.0/len(tags)
            return forward_score - gold_score,accuracy
        return forward_score - gold_score

    def forward(self, sentence):  # 与上面的 _forward_alg above 方法不同，主要用于解码.
        # 获得BiLSTM的emission scores
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        # 根据上面的lstm_feats来寻找最优路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq



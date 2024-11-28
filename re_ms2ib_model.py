import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.distributions import Normal
import torch.nn.functional as F
from model.bert import BertPreTrainedModel, BertModel
from model.agcn import TypeGraphConvolution


class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, ):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        return char_embeds

class ReMS2IB(BertPreTrainedModel):
    def __init__(self, config):
        super(ReMS2IB, self).__init__(config)
        self.bert = BertModel(config)
        self.dep_type_embedding = nn.Embedding(config.type_num, config.hidden_size, padding_idx=0)
        gcn_layer = TypeGraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn_layer = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(config.num_gcn_layers)])
        self.ensemble_linear = nn.Linear(1, config.num_gcn_layers)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*7, config.num_labels)
        self.seq_logvar_layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.seq_logvar_layer2 = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.IB = config.IB
        print(f"activate IB: {self.IB} {',beta: '+str(config.beta1)  if self.IB else ''}")
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size*2, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.char_embeddings = CharEmbeddings(config.char_num, config.hidden_size)
        self.conv1d = nn.Conv1d(config.hidden_size, config.hidden_size, 3, 1)
        self.max_pool = nn.MaxPool1d(config.max_word_length,config.max_word_length)


        self.apply(self.init_bert_weights)

    def valid_filter(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype,
                                   device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def extract_entity(self, sequence, e_mask):
        return self.max_pooling(sequence, e_mask)

    def get_attention(self, val_out, dep_embed, adj):
        batch_size, max_len, feat_dim = val_out.shape
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        val_cat = torch.cat((val_us, dep_embed), -1)
        atten_expand = (val_cat.float() * val_cat.float().transpose(1,2))
        attention_score = torch.sum(atten_expand, dim=-1)
        attention_score = attention_score / feat_dim ** 0.5
        # softmax
        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        return attention_score
    
    def _variational_layer(self, hidden, logvar_layer):
        sampled_z = hidden
        kld = torch.zeros(hidden.shape[:-1], dtype=torch.float32, device=hidden.device)
        if self.training and self.IB is True:
            mu = hidden  # 均值
            logvar = logvar_layer(hidden)  # 方差
            # TODO 训练次数设置的大点, 超5w, 8w起吧
            std = F.softplus(logvar)
            # std = torch.exp(0.5 * logvar)
            posterior = Normal(loc=mu, scale=std, validate_args=False)

            zeros = torch.zeros_like(mu, device=mu.device)
            ones = torch.ones_like(std, device=std.device)
            prior = Normal(zeros, ones, validate_args=False)

            eps = std.new_empty(std.shape)
            eps.normal_()
            sampled_z = mu + std * eps
            # (b,128)
            kld = posterior.log_prob(sampled_z).sum(-1) - prior.log_prob(sampled_z).sum(-1)

            # (b,1)
        return sampled_z, kld
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e1_mask=None, e2_mask=None,
                dep_adj_matrix=None, dep_type_matrix=None, valid_ids=None,char_sequence=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if valid_ids is not None:
            valid_sequence_output = self.valid_filter(sequence_output, valid_ids)
        else:
            valid_sequence_output = sequence_output
        x = self.dropout(valid_sequence_output)


        dep_type_embedding_outputs = self.dep_type_embedding(dep_type_matrix)
        dep_adj_matrix = torch.clamp(dep_adj_matrix, 0, 1)
        for i, gcn_layer_module in enumerate(self.gcn_layer):
            attention_score = self.get_attention(x, dep_type_embedding_outputs, dep_adj_matrix)
            sequence_output = gcn_layer_module(sequence_output, attention_score, dep_type_embedding_outputs)
        

        char_embeds = self.char_embeddings(char_sequence)
        char_embeds = char_embeds.permute(0, 2, 1)
        # print(f"embedding shape: {char_embeds.shape}") 
        char_feature = self.conv1d(char_embeds)
        # print(f"cov shape: {char_feature.shape}") 
        char_feature = self.max_pool(char_feature)
        # print(f"max pool shape: {char_feature.shape}")
        char_feature = torch.tanh(char_feature)

        char_feature = char_feature.permute(0, 2, 1)
        # print(f"final shape: {char_feature.shape}")
        


        x = self.transformer_encoder(torch.cat([x,char_feature],-1))
        # print(f"x shape {x.shape}")
        
        x,kld2 = self._variational_layer(x,self.seq_logvar_layer2)
        sequence_output,kld1 = self._variational_layer(sequence_output,self.seq_logvar_layer1)
        l1 = torch.matmul(kld1, (e1_mask + e2_mask).T.type(torch.float32)).sum()
        l2 = torch.matmul(kld2, (e1_mask + e2_mask).T.type(torch.float32)).sum()
        sequence_output = torch.cat([sequence_output,x],dim=-1)
        # dropout
        sequence_output = self.dropout(sequence_output)
        e1_h = self.extract_entity(sequence_output, e1_mask)
        e2_h = self.extract_entity(sequence_output, e2_mask)

        pooled_output = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))+l1*self.beta1+l2*self.beta2
            return loss
        else:
            return logits

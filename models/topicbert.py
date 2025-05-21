import torch
import torch.nn as nn
from transformers import ModernBertModel
from models.smdirichlet import SMDIRICHLET


class TopicBERT(nn.Module):
    '''Implementation of the TopicBERT model with co-attention between BERT and topic embeddings.'''
    def __init__(self, vocab_size, num_labels, alpha=0.9, dropout=0.1):
        super().__init__()
        self.alpha = alpha
        self.encoder = ModernBertModel.from_pretrained('answerdotai/ModernBERT-base')
        self.smdirichlet = SMDIRICHLET(vocab_size)
        #self.softmax = nn.Softmax(dim=-1)

        # Co-attention projection layers
        hidden_size = self.encoder.config.hidden_size
        topic_dim = self.smdirichlet.num_topics
        self.co_attn_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.co_attn_t = nn.Linear(topic_dim, hidden_size, bias=False)
        self.attn_bias = nn.Parameter(torch.zeros(1))

        # Combine co-attended representation
        self.combine_proj = nn.Linear(hidden_size, hidden_size)

        # Classification head
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels)
        )
        self.projection.apply(TopicBERT._get_init_transformer(self.encoder))

        self.bert_loss = nn.CrossEntropyLoss(reduction='mean')

    @staticmethod
    def _get_init_transformer(transformer):
        def init_transformer(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=transformer.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        return init_transformer

    def forward(self, input_ids, attention_mask, bows, labels):
        # BERT encoding
        hiddens_last = self.encoder(input_ids, attention_mask=attention_mask)[0]
        embs = hiddens_last[:, 0, :]  # [CLS] token embeddings

        # Topic model encoding
        h_tm, _, kld, loss_diri = self.smdirichlet(bows)

        # Co-attention
        proj_b = self.co_attn_b(embs)            # (batch_size, hidden_size)
        proj_t = self.co_attn_t(h_tm)           # (batch_size, hidden_size)
        # Compute affinity and attention weight
        scores = torch.sum(proj_b * proj_t, dim=1, keepdim=True) + self.attn_bias  # (batch_size, 1)
        alpha = torch.sigmoid(scores)           # (batch_size, 1)
        #alpha = self.softmax(scores)
        # Fuse representations
        joint = alpha * proj_b + (1 - alpha) * proj_t
        co_emb = torch.tanh(self.combine_proj(joint))  # (batch_size, hidden_size)

        # Classification
        logits = self.projection(co_emb)

        # Loss computation
        loss_bert = self.bert_loss(logits, labels.max(1).indices)
        loss_total = loss_bert + kld * 0.0001
        return logits, loss_total, kld
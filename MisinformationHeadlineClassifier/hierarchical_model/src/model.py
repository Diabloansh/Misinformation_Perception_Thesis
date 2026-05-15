import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional

class HeadlineClassifier(nn.Module):

    def __init__(self, model_name: str, num_labels: int=3, dropout_rate: float=0.1, class_weights: Optional[torch.Tensor]=None):
        super().__init__()
        self.num_labels = num_labels
        self.bert_config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert_config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor]=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # use the CLS token for the final classifier
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}

def build_model(config, class_weights: Optional[torch.Tensor]=None) -> HeadlineClassifier:
    return HeadlineClassifier(model_name=config.model_name, num_labels=config.num_labels, dropout_rate=getattr(config, 'dropout_rate', 0.1), class_weights=class_weights)

from typing import Callable
import torch
from torch.nn.functional import linear, normalize

class PartialFC_V2_DP(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.sample_rate = sample_rate
        self.fp16 = fp16
        self.num_local = num_classes
        self.num_sample = int(self.sample_rate * self.num_local)
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))
        self.margin_softmax = margin_loss

    def sample(self, labels, index_positive):
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).cuda(non_blocking=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local]).cuda(non_blocking=True)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.weight_index = index
            labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        return self.weight[self.weight_index]

    def forward(self, embeddings, labels):
        labels = labels.long().view(-1, 1)
        index_positive = (labels >= 0)

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        with torch.amp.autocast("cuda", enabled=self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight = normalize(weight)
            logits = linear(norm_embeddings, norm_weight)
            logits = logits.clamp(-1, 1)
            logits = self.margin_softmax(logits, labels.squeeze())
            loss = torch.nn.functional.cross_entropy(logits, labels.squeeze())
        return loss

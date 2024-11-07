from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from arch_eval import Model, ClassificationModel

# implement a child class of Model
class CodecWrapper(Model):
    def __init__(self, model, device, max_length, train_backbone=False):
        super().__init__(model)
        self.model = model
        # the model must not be trained
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.train_backbone = train_backbone
        self.sampling_rate = 24_000
        self.token_embedding_size = 512

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        audio = audio.unsqueeze(0)  # [1, T]
        audio = audio.to(self.device)
        bandwidth_id = torch.tensor([0])
        if self.train_backbone:
            token_embeddings,discrete_code= self.model.encode_infer(audio, bandwidth_id=bandwidth_id)   # [1, D, T]
        else:
            with torch.no_grad():
                token_embeddings,discrete_code= self.model.encode_infer(audio, bandwidth_id=bandwidth_id)

        embeddings = token_embeddings.mean(dim=-1).squeeze()
        # move the embeddings to the cpu
        embeddings = embeddings.cpu()
        return embeddings

    # def get_sequence_embeddings(self, audio: np.ndarray, **kwargs):
    #     audio = audio.squeeze(0)
    #     audio = audio.to(self.device)
    #     bandwidth_id = torch.tensor([0])
    #     if self.train_backbone:
    #         token_embeddings,discrete_code= self.model.encode_infer(audio, bandwidth_id=bandwidth_id)
    #     else:
    #         with torch.no_grad():
    #             token_embeddings,discrete_code= self.model.encode_infer(audio, bandwidth_id=bandwidth_id)

    #     # move the embeddings to the cpu
    #     if self.train_backbone:
    #         token_embeddings = token_embeddings.cpu()
    #     else:
    #         with torch.no_grad():
    #             token_embeddings = token_embeddings.detach().cpu()

    #     return token_embeddings.squeeze()

    def get_classification_embedding_size(self):
        return self.token_embedding_size

    # def get_token_embedding_size(self):
    #     return self.token_embedding_size

    def get_sampling_rate(self):
        return self.sampling_rate
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .wan_2_2.modules.s2v.audio_encoder import get_sample_indices,linear_interpolation,AudioEncoder

class AudioEncoder_Training(AudioEncoder): #和causal 没关系，只是训练时需要把 processor 放在 dataloader 里， 需要重构下原始encoder 类
    def __init__(self, device='cpu', model_id="facebook/wav2vec2-base-960h"):
        # load pretrained model
        # self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)

        self.model = self.model.to(device)

        self.video_rate = 30

    def extract_audio_feat_training(self,
                           audio_tensor,
                           return_all_layers=False,
                           dtype=torch.float32):

        # Training
        # retrieve logits & take argmax
        res = self.model(
            audio_tensor.to(self.model.device), output_hidden_states=True)
        if return_all_layers:
            feat = torch.cat(res.hidden_states)
        else:
            feat = res.hidden_states[-1]
        feat = linear_interpolation(
            feat, input_fps=50, output_fps=self.video_rate)

        z = feat.to(dtype)  # Encoding for the motion
        return z

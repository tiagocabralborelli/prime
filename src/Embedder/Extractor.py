import pandas as pd
import torch
import numpy as np
from contextlib import nullcontext
from sklearn.base import clone
from transformers import EsmTokenizer, EsmForMaskedLM

try:
    from peft import PeftConfig, PeftModel
except ImportError:
    PeftConfig = None
    PeftModel = None

class EmbeddingExtractor:
    def __init__(self, path, device):
        self.device = torch.device(device)
        self.check_point = path

    def loadModel(self, tokenizer, model_loader):
        self._tokenizer = tokenizer.from_pretrained(self.check_point)
        self._base_model = model_loader.from_pretrained(self.check_point)
        self.max_len = self._base_model.config.max_position_embeddings

class EmbedderESM(EmbeddingExtractor):
    def loadModel(self, tokenizer, model_loader,  name = 'ESM2-Vanila'):
        super().loadModel(tokenizer, model_loader)
        self._base_model = self._base_model.to(self.device)
        self._base_model.eval()

    def embedding(self, sequences, batch_size = 128):
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = self._tokenizer(batch, return_tensors='pt', padding=True,
                                    truncation=True, max_length=self.max_len).to(self.device)
            with torch.no_grad():
                autocast_context = torch.cuda.amp.autocast() if self.device.type == 'cuda' else nullcontext()
                with autocast_context:
                    outputs = self._base_model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    mask = inputs.attention_mask.unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
        self._embeddings =  np.vstack(embeddings)
        return self._embeddings

class EmbedderLoRAESM(EmbeddingExtractor):
    def __init__(self, path, device):
        super().__init__(path, device)
        self.base_model_name = path

    def loadModel(self, tokenizer=EsmTokenizer, model_loader=EsmForMaskedLM, name='ESM2-LoRA'):
        self._tokenizer = tokenizer.from_pretrained(self.check_point)

        adapter_config = None
        if PeftConfig is not None:
            try:
                adapter_config = PeftConfig.from_pretrained(self.check_point)
            except Exception:
                adapter_config = None

        if adapter_config is not None:
            base_model_name = adapter_config.base_model_name_or_path or self.base_model_name
            base_model = model_loader.from_pretrained(base_model_name)
            self._base_model = PeftModel.from_pretrained(base_model, self.check_point)
        else:
            self._base_model = model_loader.from_pretrained(self.check_point)

        self.max_len = self._base_model.config.max_position_embeddings
        self._base_model = self._base_model.to(self.device)
        self._base_model.eval()

        if name is not None:
            self.model_name = name

    def embedding(self, sequences, batch_size = 128):
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = self._tokenizer(batch, return_tensors='pt', padding=True,
                                    truncation=True, max_length=self.max_len).to(self.device)
            with torch.no_grad():
                autocast_context = torch.cuda.amp.autocast() if self.device.type == 'cuda' else nullcontext()
                with autocast_context:
                    outputs = self._base_model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    mask = inputs.attention_mask.unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
        self._embeddings =  np.vstack(embeddings)
        return self._embeddings
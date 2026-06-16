import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MultiLabelDataset(Dataset):
    """Single-label text dataset with cached BERT features."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bert_features = [None] * len(texts)

    def __len__(self):
        return len(self.texts)

    def cache_bert_features(self, model, device):
        model.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(self)), desc="Caching BERT features"):
                encoding = self.tokenizer.encode_plus(
                    str(self.texts[idx]),
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                self.bert_features[idx] = outputs.last_hidden_state[:, 0, :].cpu()

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        bert_feature = self.bert_features[idx]
        if bert_feature is None:
            bert_feature = torch.zeros(768, dtype=torch.float32)
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "bert_features": bert_feature.flatten(),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

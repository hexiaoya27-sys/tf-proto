import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from config.Config import Config
from data.dataset import MultiLabelDataset
from models.meta_model import MultiModalModel
from utils.maml import MAML
from utils.task_utils import create_task_batch


def load_d2_json(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            label = int(obj["label"])
            if 0 <= label <= 29:
                text = obj.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                texts.append(str(text))
                labels.append(label)
    return texts, labels


def evaluate(model, dataloader, device, n_labels):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bert_features = batch["bert_features"].to(device)
            labels = batch["labels"].to(device)
            _, logits = model(input_ids, attention_mask, bert_features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(n_labels)),
        target_names=[f"Label_{i}" for i in range(n_labels)],
        digits=4,
        zero_division=0,
    )
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    return report, accuracy


def main_training_loop(config, texts, labels):
    device = torch.device(config.DEVICE)
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)
    feature_model = BertModel.from_pretrained(config.MODEL_PATH).to(device)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels,
    )

    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer, config.MAX_LEN)
    train_dataset.cache_bert_features(feature_model, device)
    val_dataset.cache_bert_features(feature_model, device)
    del feature_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = MultiModalModel(n_labels=config.N_LABELS, model_path=config.MODEL_PATH).to(device)
    maml = MAML(model, inner_lr=config.INNER_LR, outer_lr=config.OUTER_LR)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.STANDARD_LR)

    best_val_acc = -1.0
    best_path = Path(config.MODEL_SAVE_PATH)

    for epoch in range(1, config.EPOCHS + 1):
        task_batch = create_task_batch(
            train_dataset,
            n_way=min(config.N_WAY, config.N_LABELS),
            k_shot=config.K_SHOT,
            query_size=config.QUERY_SIZE,
            num_tasks=config.NUM_TASKS,
        )
        maml_loss = float(maml.step(task_batch, device)) if task_batch else 0.0

        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bert_features = batch["bert_features"].to(device)
            labels_batch = batch["labels"].to(device)
            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, bert_features, labels_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_report, val_acc = evaluate(model, val_loader, device, config.N_LABELS)
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print(f"MAML Loss: {maml_loss:.4f} | Train Loss: {np.mean(train_losses):.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(val_report)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_report, val_acc = evaluate(model, val_loader, device, config.N_LABELS)
    print("\n=== Final Evaluation ===")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(val_report)


if __name__ == "__main__":
    cfg = Config()
    d2_path = PROJECT_ROOT / "data" / "cyber-violence_cn.json"
    texts_, labels_ = load_d2_json(d2_path)
    main_training_loop(cfg, texts_, labels_)

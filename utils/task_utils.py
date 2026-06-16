import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from config.Config import Config

def create_task_batch(dataset, n_way=5, k_shot=1, query_size=5, num_tasks=10):
    """为元学习创建任务批次"""
    task_batch = []
    label_set = list(range(30))  # 标签0-4

    for _ in range(num_tasks):
        # 随机选择n_way个类别
        selected_labels = np.random.choice(label_set, n_way, replace=False)

        # 为每个类别选择支持集和查询集样本
        support_set = {'input_ids': [], 'attention_mask': [], 'bert_features': [], 'labels': []}
        query_set = {'input_ids': [], 'attention_mask': [], 'bert_features': [], 'labels': []}

        for label in selected_labels:
            # 获取该标签的所有样本
            indices = [i for i, lbl in enumerate(dataset.labels) if lbl == label]

            if len(indices) < k_shot + query_size:
                continue

            # 随机选择样本
            selected_indices = np.random.choice(indices, k_shot + query_size, replace=False)

            # 添加到支持集和查询集
            for i, idx in enumerate(selected_indices):
                sample = dataset[idx]
                if i < k_shot:  # 支持集
                    for key in ['input_ids', 'attention_mask', 'bert_features', 'labels']:
                        support_set[key].append(sample[key])
                else:  # 查询集
                    for key in ['input_ids', 'attention_mask', 'bert_features', 'labels']:
                        query_set[key].append(sample[key])

        if len(support_set['labels']) == 0:
            continue

        # 转换为张量
        for key in ['input_ids', 'attention_mask', 'bert_features']:
            if len(support_set[key]) > 0:
                support_set[key] = torch.stack(support_set[key])
                query_set[key] = torch.stack(query_set[key])
        support_set['labels'] = torch.stack(support_set['labels'])
        query_set['labels'] = torch.stack(query_set['labels'])

        task_batch.append((support_set, query_set))

    return task_batch


def evaluate(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bert_features = batch['bert_features'].to(device)
            labels = batch['labels'].to(device)

            _, logits = model(input_ids, attention_mask, bert_features)

            # 单标签分类
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    # 计算分类报告
    report = classification_report(
        all_targets, all_preds,
        target_names=[f'Label_{i}' for i in range(30)],
        digits=4
    )

    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    return report, accuracy
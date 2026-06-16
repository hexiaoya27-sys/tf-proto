from torch.optim import AdamW
from torch.optim import AdamW
import torch

class MAML:
    """MAML算法实现（元学习优化器）"""

    def __init__(self, model, inner_lr=0.01, outer_lr=1e-4, adaptation_steps=1):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.adaptation_steps = adaptation_steps
        self.optimizer = AdamW(model.parameters(), lr=outer_lr)

    def step(self, task_batch, device):
        """执行一个MAML训练步骤"""
        self.optimizer.zero_grad()
        total_loss = 0

        for support_set, query_set in task_batch:
            # 1. 在支持集上计算损失
            self.model.train()
            loss, _ = self.model(
                input_ids=support_set['input_ids'].to(device),
                attention_mask=support_set['attention_mask'].to(device),
                bert_features=support_set['bert_features'].to(device),
                labels=support_set['labels'].to(device)
            )

            # 2. 计算梯度并更新快速权重
            params = [p for p in self.model.parameters() if p.requires_grad]
            gradients = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

            # 3. 创建快速权重
            fast_weights = []
            for param, grad in zip(params, gradients):
                if grad is not None:
                    fast_weights.append(param - self.inner_lr * grad)
                else:
                    fast_weights.append(param)

            # 4. 在查询集上计算损失
            original_params = [p.clone().detach() for p in params]
            for param, new_param in zip(params, fast_weights):
                param.data.copy_(new_param.data)

            query_loss, _ = self.model(
                input_ids=query_set['input_ids'].to(device),
                attention_mask=query_set['attention_mask'].to(device),
                bert_features=query_set['bert_features'].to(device),
                labels=query_set['labels'].to(device)
            )

            # 5. 恢复原始参数
            for param, orig_param in zip(params, original_params):
                param.data.copy_(orig_param.data)

            # 6. 反向传播
            if query_loss != 0:
                query_loss.backward(retain_graph=True)
            total_loss += query_loss.item() if query_loss != 0 else 0

        # 平均损失并更新原始参数
        if len(task_batch) > 0:
            total_loss /= len(task_batch)
            self.optimizer.step()

        return total_loss
# utils/maml.py
# from torch.optim import AdamW
# import torch
# from collections import OrderedDict

# class MAML:
#     """MAML算法实现（元学习优化器）
#     - 新增: update_bert 开关，控制内环是否更新 BERT/text 分支
#     - 新增: allow_prefixes / forbid_keywords，精细控制可更新参数的范围
#     """
#
#     def __init__(
#         self,
#         model,
#         inner_lr=0.01,
#         outer_lr=1e-4,
#         adaptation_steps=1,
#         *,
#         update_bert=False,                          # 是否让内环更新 BERT
#         allow_prefixes=("fusion", "classifier", "cls_head", "proj", "fc"),
#         bert_keywords=("online_bert", "text_encoder", "bert", "encoder"),
#         forbid_keywords=(),                         # 额外禁止关键词（如 "bn", "layer_norm"）
#     ):
#         self.model = model
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.adaptation_steps = adaptation_steps
#         self.update_bert = bool(update_bert)
#         self.allow_prefixes = tuple(allow_prefixes) if allow_prefixes else ()
#         self.bert_keywords = tuple(bert_keywords)
#         self.forbid_keywords = tuple(forbid_keywords) if forbid_keywords else ()
#
#         self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=outer_lr)
#
#     # ---- 参数筛选策略 ----
#     def _is_bert_param(self, name: str) -> bool:
#         return any(k in name for k in self.bert_keywords)
#
#     def _is_forbidden(self, name: str) -> bool:
#         if any(k in name for k in self.forbid_keywords):
#             return True
#         if not self.update_bert and self._is_bert_param(name):
#             return True
#         return False
#
#     def _is_allowed_by_prefix(self, name: str) -> bool:
#         if not self.allow_prefixes:
#             return True
#         for p in self.allow_prefixes:
#             if name.startswith(p) or f".{p}." in name or name.endswith(p):
#                 return True
#         if self.update_bert and self._is_bert_param(name):
#             return True
#         return False
#
#     def _iter_adaptable_params(self):
#         pairs = []
#         for n, p in self.model.named_parameters():
#             if not p.requires_grad:          # 尊重 requires_grad
#                 continue
#             if self._is_forbidden(n):
#                 continue
#             if not self._is_allowed_by_prefix(n):
#                 continue
#             pairs.append((n, p))
#         return pairs
#
#     def step(self, task_batch, device):
#         """执行一个MAML训练步骤（支持 adaptation_steps 次内环更新）"""
#         self.optimizer.zero_grad()
#         total_loss_val, num_tasks = 0.0, 0
#
#         for support_set, query_set in task_batch:
#             num_tasks += 1
#             adaptable = self._iter_adaptable_params()
#             params_only = [p for _, p in adaptable]
#             original_params = [p.clone().detach() for p in params_only]
#
#             try:
#                 fast_params = params_only
#                 for _ in range(max(1, int(self.adaptation_steps))):
#                     self.model.train()
#                     loss_s, _ = self.model(
#                         input_ids=support_set['input_ids'].to(device),
#                         attention_mask=support_set['attention_mask'].to(device),
#                         bert_features=support_set['bert_features'].to(device),
#                         labels=support_set['labels'].to(device)
#                     )
#                     grads = torch.autograd.grad(loss_s, fast_params, create_graph=True, allow_unused=True)
#                     new_fast = []
#                     for p, g in zip(fast_params, grads):
#                         new_fast.append(p - self.inner_lr * g if g is not None else p)
#                     fast_params = new_fast
#                     for (_, p), new_p in zip(adaptable, fast_params):
#                         p.data.copy_(new_p.data)
#
#                 self.model.train()
#                 loss_q, _ = self.model(
#                     input_ids=query_set['input_ids'].to(device),
#                     attention_mask=query_set['attention_mask'].to(device),
#                     bert_features=query_set['bert_features'].to(device),
#                     labels=query_set['labels'].to(device)
#                 )
#
#             finally:
#                 for p, orig in zip(params_only, original_params):
#                     p.data.copy_(orig.data)
#
#             if loss_q is not None and loss_q != 0:
#                 loss_q.backward()
#                 total_loss_val += float(loss_q.item())
#
#         if num_tasks > 0:
#             total_loss_val /= num_tasks
#             self.optimizer.step()
#
#         return total_loss_val
#

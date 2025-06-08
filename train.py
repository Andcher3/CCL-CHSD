import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import os
import time  # 用于记录训练时间

# 导入你之前定义好的模块
from Hype import *
from data_extractor import load_data
from data_formatter import convert_samples_to_features, tokenizer
from data_loader import CHSDDataset, collate_fn
from Model import HateSpeechDetectionModel  # 你的模型
from eval import evaluate_model
# 导入即将创建的损失计算函数
from loss import compute_total_loss  # 假设这个函数将在 loss_calculator.py 中定义

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_epoch(model, dataloader, optimizer, scheduler, epoch=EPOCH):
    """
    main training loop
    :param model: chsd model
    :param dataloader: dataloader from data_loader
    :param optimizer: optimizer
    :param scheduler: use with optimizer
    :param epoch: EPOCH from Hype
    :return: avg_epoch_loss
    """

    model.train()  # 设置模型为训练模式
    total_loss_sum = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()  # 清零梯度

        # 将数据移动到指定设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        quads_labels_batch = batch['quads_labels']  # 这是一个列表的列表，需要逐样本处理

        # 1. 模型前向传播
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        # 2. 计算损失 (调用外部函数)
        # compute_total_loss 将接收模型输出和真实标签，返回总损失以及各组件损失
        # 我们将在 loss_calculator.py 中实现这个函数
        loss, loss_components = compute_total_loss(
            outputs=outputs,
            quads_labels_batch=quads_labels_batch,
            model=model,  # 传入模型实例以便在loss_calculator中调用classify_quad
            device=device,
            span_weight=1.0,
            group_weight=0.6,
            hateful_weight=0.8,
            biaffine_weight=0.4
            # 可以传入权重，例如 span_weight=1.0, group_weight=0.5, hateful_weight=0.5
        )

        # 3. 反向传播与参数更新
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss_sum += loss.item()

        # 打印训练进度 (每10个 batch 打印一次)
        if (batch_idx + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_batch_time = elapsed_time / (batch_idx + 1)
            remaining_time = avg_batch_time * (len(dataloader) - (batch_idx + 1))

            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Total Loss: {loss.item():.4f} | "
                  f"IOU/KL Span: {loss_components.get('iou_span_loss', 0.0):.4f}/{loss_components.get('kl_span_loss', 0.0):.3f} | "  # 使用.get()确保即使组件不存在也不会报错
                  f"Biaffine: {loss_components.get('biaffine_loss', 0.0):.4f} | "
                  f"Group: {loss_components.get('group_loss', 0.0):.4f} | "
                  f"Hateful: {loss_components.get('hateful_loss', 0.0):.4f} | "
                  f"Time: {avg_batch_time:.2f}s/batch | Est. Remaining: {remaining_time:.0f}s")
            # 重置计时器，使每次打印的时间更准确反映最近的 batch
            # start_time = time.time() # 如果想统计每10个batch的平均时间，可以取消注释

    avg_epoch_loss = total_loss_sum / len(dataloader)
    print(f"\n--- Epoch {epoch + 1} Summary ---")
    print(f"Average Total Loss for Epoch: {avg_epoch_loss:.4f}")
    print(f"Epoch Training Time: {time.time() - start_time:.2f} seconds")

    return avg_epoch_loss


def main():
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    # 1. 数据加载与预处理
    print("Loading and preprocessing data...")
    train_path = os.path.join("data", "train.json")
    processed_train_data, processed_test_data = load_data(train_path, split_ratio=0.3)
    train_features = convert_samples_to_features(processed_train_data)
    train_dataset = CHSDDataset(train_features)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    test_features = convert_samples_to_features(processed_test_data)
    test_dataset = CHSDDataset(test_features)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Training data loaded. {len(train_dataloader)} batches, {len(processed_train_data)} samples.")
    print(f"Testing data loaded. {len(test_dataloader)} batches, {len(processed_test_data)} samples.")

    # 2. 模型实例化
    print("Initializing model...")
    model = HateSpeechDetectionModel(bert_model_name='bert-base-chinese').to(device)
    print("Model initialized.")

    # 3. 优化器和学习率调度器
    print("Setting up optimizer and scheduler...")
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)

    total_steps = len(train_dataloader) * EPOCH  # batch_num * epoch = total_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.1,
                                                num_training_steps=total_steps)
    print(f"Optimizer and scheduler set up. Total steps: {total_steps}")

    # 4. 训练循环
    print("\n--- Starting Training ---")
    best_avg_f1 = 0
    for epoch in range(EPOCH):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch)

        metrics = evaluate_model(model, test_dataloader, tokenizer, device)
        print(f"Validation Hard F1: {metrics['hard_f1']:.4f}")
        print(f"Validation Soft F1: {metrics['soft_f1']:.4f}")
        print(f"Validation Average F1: {metrics['average_f1']:.4f}")

        if metrics['average_f1'] > best_avg_f1:
            best_avg_f1 = metrics['average_f1']
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pth"))

        # 可以在这里添加验证集的评估、模型保存等逻辑
        if (epoch + 1) % SAVE_EPOCH == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch + 1}.pth"))

        # torch.cuda.empty_cache()

    print("\n--- Training Finished ---")
    # 最终模型保存
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "final_model.pth"))


if __name__ == '__main__':
    main()

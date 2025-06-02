import torch
from Model import HateSpeechDetectionModel  # 假设你的模型定义在 model.py 中
from Hype import TARGET_GROUP_CLASS_NAME, MAX_SEQ_LENGTH  # 假设你的参数定义在 Hype.py 中


def test_model_forward():
    print("--- 开始测试模型 forward 方法 ---")

    # 1. 定义测试参数
    batch_size = 2
    seq_len = MAX_SEQ_LENGTH  # 从 Hype.py 获取最大序列长度

    # 2. 实例化模型
    # 注意：这里我们使用 'bert-base-chinese'，首次运行时会下载模型权重
    model = HateSpeechDetectionModel(bert_model_name='bert-base-chinese')
    model.eval()  # 将模型设置为评估模式，禁用 dropout 等

    # 获取 BERT 模型的实际词汇表大小和隐藏层维度
    vocab_size = model.bert.config.vocab_size
    hidden_size = model.bert.config.hidden_size

    print(f"BERT model's vocab_size: {vocab_size}")
    print(f"BERT model's hidden_size: {hidden_size}")

    # 3. 创建 Dummy Input
    # input_ids: (batch_size, seq_len) - 模拟 token IDs
    # 修正: 使用实际的 vocab_size 作为随机整数的上限
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # attention_mask: (batch_size, seq_len) - 模拟注意力掩码 (1 for real tokens, 0 for padding)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # token_type_ids: (batch_size, seq_len) - 模拟 segment IDs (对于单句任务通常全0)
    dummy_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    print(f"Dummy Input Shapes:")
    print(f"  input_ids: {dummy_input_ids.shape}")
    print(f"  attention_mask: {dummy_attention_mask.shape}")
    print(f"  token_type_ids: {dummy_token_type_ids.shape}")

    # 4. 执行 forward pass
    with torch.no_grad():  # 在测试时禁用梯度计算
        outputs = model(input_ids=dummy_input_ids,
                        attention_mask=dummy_attention_mask,
                        token_type_ids=dummy_token_type_ids)

    # 5. 检查输出形状
    print("\nModel Output Shapes:")

    # Span Extraction Logits
    target_start_logits_shape = outputs['target_start_logits'].shape
    target_end_logits_shape = outputs['target_end_logits'].shape
    argument_start_logits_shape = outputs['argument_start_logits'].shape
    argument_end_logits_shape = outputs['argument_end_logits'].shape

    print(f"  target_start_logits: {target_start_logits_shape} (Expected: ({batch_size}, {seq_len}, 1))")
    print(f"  target_end_logits: {target_end_logits_shape} (Expected: ({batch_size}, {seq_len}, 1))")
    print(f"  argument_start_logits: {argument_start_logits_shape} (Expected: ({batch_size}, {seq_len}, 1))")
    print(f"  argument_end_logits: {argument_end_logits_shape} (Expected: ({batch_size}, {seq_len}, 1))")

    # BERT Sequence Output
    sequence_output = outputs['sequence_output']  # 直接从 outputs 获取
    sequence_output_shape = sequence_output.shape
    print(f"  sequence_output: {sequence_output_shape} (Expected: ({batch_size}, {seq_len}, {hidden_size}))")

    # BERT CLS Output (pooled_output)
    cls_output = outputs['cls_output']  # 直接从 outputs 获取
    cls_output_shape = cls_output.shape
    print(f"  cls_output: {cls_output_shape} (Expected: ({batch_size}, {hidden_size}))")

    # 6. 验证 BiaffinePairing 层 (单独测试)
    print("\n--- 测试 BiaffinePairing 层 ---")
    # 假设我们从 sequence_output 中为第一个样本提取了 3 个 target spans 和 4 个 argument spans
    # 更准确地模拟：从实际的 sequence_output 中切片来创建
    target_spans_for_sample1 = sequence_output[0, :3, :].clone()  # 模拟从第一个样本提取3个spans
    argument_spans_for_sample1 = sequence_output[0, 5:9, :].clone()  # 模拟从第一个样本提取4个spans

    biaffine_scores_sample1 = model.biaffine_pairing(target_spans_for_sample1, argument_spans_for_sample1)
    print(f"  Biaffine scores for one sample: {biaffine_scores_sample1.shape} (Expected: ({3}, {4}))")

    # 7. 验证分类头 (classify_quad 方法)
    print("\n--- 测试 classify_quad 方法 ---")
    # 假设我们为第一个样本获取了某个 quad 的 token-level span 索引
    # 这些索引需要是真实的，或者有效的dummy值，以确保_get_span_representation能工作

    # 沿用上面获取的 sequence_output 和 cls_output
    # 假设一个有效的 span 索引
    # 例如：target span: token 1-3, argument span: token 5-7 (这些索引必须在 seq_len 范围内)
    dummy_t_start, dummy_t_end = 1, 3
    dummy_a_start, dummy_a_end = 5, 7
    dummy_batch_idx = 0  # 对于测试第一个样本，batch_idx为0

    # 修正：classify_quad 方法的调用
    group_logits, hateful_logits = model.classify_quad(
        sequence_output=sequence_output,  # 传入整个 batch 的 sequence_output
        cls_output=cls_output,  # 传入整个 batch 的 cls_output
        t_start_token=dummy_t_start, t_end_token=dummy_t_end,
        a_start_token=dummy_a_start, a_end_token=dummy_a_end,
        batch_idx=dummy_batch_idx  # 指明是 batch 中的第几个样本
    )

    print(f"  Group logits (classify_quad): {group_logits.shape} (Expected: (1, {len(TARGET_GROUP_CLASS_NAME)}))")
    print(f"  Hateful logits (classify_quad): {hateful_logits.shape} (Expected: (1, 1))")

    print("\n--- 模型 forward 方法测试通过！ ---")


if __name__ == '__main__':
    test_model_forward()
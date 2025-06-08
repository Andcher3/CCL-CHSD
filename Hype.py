
# target groups name
TARGET_GROUP_CLASS_NAME = ['Region', 'Racism', 'Sexism', 'LGBTQ', 'others', 'non-hate']
MODEL_NAME = "hfl/chinese-macbert-large"
# MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
# MODEL_NAME = "hfl/chinese-bert-base"

# max original text length
MAX_SEQ_LENGTH = 128

# span select stradegy
TOPK_SPAN = 8
MAX_SPAN_LENGTH = 15
BOUNDARY_SMOOTHING_EPSILON = 0.85
BOUNDARY_SMOOTHING_D = 1
EPSILON = 0.05
D_SMOOTH = 1
MAX_CANDIDATE_SPANS_PER_SAMPLE_FOR_KL_LOSS = 500
# training args
LR = 2e-5
BATCH_SIZE = 16
EPOCH = 100

# saving args
SAVE_EPOCH = 10
MODEL_SAVE_PATH = "results/model"


# eval related
PAIRING_THRESHOLD = 0.5
NMS_OVERLAP_THRESHOLD = 0.5 # NMS 的 IoU 重叠阈值 (需要调优)

# --- NMS Configuration ---
APPLY_NMS = True
NMS_IOU_THRESHOLD_TARGET = 0.7  # IoU threshold for target spans
NMS_IOU_THRESHOLD_ARGUMENT = 0.7  # IoU threshold for argument spans
NMS_CONTAINMENT_THRESHOLD = 0.85  # Min portion of one span covered by another to be considered similar

if __name__ == '__main__':
    import torch
    print(torch.cuda.is_available())
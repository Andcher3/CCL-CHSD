
# target groups name
TARGET_GROUP_CLASS_NAME = ['Region', 'Racism', 'Sexism', 'LGBTQ', 'others', 'non-hate']

# max original text length
MAX_SEQ_LENGTH = 128

# span select stradegy
TOPK_SPAN = 8
MAX_SPAN_LENGTH = 15

# training args
LR = 2e-5
BATCH_SIZE = 16
EPOCH = 50

# saving args
SAVE_EPOCH = 10
MODEL_SAVE_PATH = "results/model"

# eval related
PAIRING_THRESHOLD = 0.5

if __name__ == '__main__':
    import torch
    print(torch.cuda.is_available())
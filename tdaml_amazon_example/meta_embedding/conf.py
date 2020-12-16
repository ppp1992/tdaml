# import torch

# device = torch.device("cuda")
device = None
cvr_flag = None
model_type = None

table_path = None

model_type_params = {
    'DeepFM': {
        'base_lr': 1e-9,
        'test_lr': 1e-4,
        'train_lr': 5e-5,
        'alpha': 0.1,
        'penalty': 1e-7
    },
    'DNN': {
        'base_lr': 1e-11,
        'test_lr': 5e-4,
        'train_lr': 4e-5,
        'alpha': 0.1,
        'penalty': 1e-5,
    },
    'WD': {
        'base_lr': 1e-6,
        'test_lr': 7e-4,
        'train_lr': 5e-5,
        'alpha': 0.9,
        'penalty': 1e-6,
    },
    'FM': {
        'base_lr': 1e-6,
        'test_lr': 5e-4,
        'train_lr': 1e-4,
        'alpha': 0.1,
        'penalty': 1e-6,
    },
}

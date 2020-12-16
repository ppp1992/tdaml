# import torch

# device = torch.device("cuda")
device = None
test_device = 'cuda:3'
cvr_flag = None
model_type = None

table_path = None

model_type_params = {
    'DeepFM': {
        'base_lr': 1e-7,
        'base_penalty': 0,

        'train_lr': 1e-4,
        'alpha': 0.1,
        'penalty': 0,

        'test_lr': 1e-3,
        'test_lr1': 1e-2,
        'test_penalty': 1e-6,
        'test_penalty1': 1e-6,
    },
    'DNN': {
        'base_lr': 1e-6,
        'base_penalty': 1e-7,

        'train_lr': 1e-4,
        'alpha': 0.1,
        'penalty': 0,

        'test_lr': 1e-3,
        'test_lr1': 9e-3,
        'test_penalty': 1e-6,
        'test_penalty1': 1e-6,
    },
    'WD': {
        'base_lr': 1e-6,
        'base_penalty': 1e-7,

        'train_lr': 1e-5,
        'alpha': 0.1,
        'penalty': 0,

        'test_lr': 1e-4,
        'test_lr1': 1e-3,
        'test_penalty': 1e-6,
        'test_penalty1': 1e-5,
    },
    'FM': {
        'base_lr': 1e-7,
        'base_penalty': 1e-9,

        'train_lr': 1e-4,
        'alpha': 0.1,
        'penalty': 0,

        'test_lr': 5e-4,
        'test_lr1': 1e-3,
        'test_penalty': 1e-6,
        'test_penalty1': 1e-6,
    },
}

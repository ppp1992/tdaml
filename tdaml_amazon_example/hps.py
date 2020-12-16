from hyperopt import hp

models = {
    'DNN': {
        'lr': hp.choice('lr', [0.0001]),
        'weight_decay': hp.choice('weight_decay', [1e-6]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [25]),
        'rho': hp.choice('rho', [1e-08]),
        'p_lr': hp.choice('p_lr', [0.01]),
        'p_lr_decay': hp.choice('p_lr_decay', [1.0]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.1]),
        'rest': hp.choice('rest', [True]),
    },

    'WD': {
        'lr': hp.choice('lr', [0.001]),
        'weight_decay': hp.choice('weight_decay', [1e-7]),
        'amsgrad': hp.choice('amsgrad', [True]),
        'batch_n_ID': hp.choice('batch_n_ID', [50]),
        'rho': hp.choice('rho', [1e-9]),
        'p_lr': hp.choice('p_lr', [0.0001]),
        'p_lr_decay': hp.choice('p_lr_decay', [0.01]),
        'gamma': hp.choice('gamma', [0.5]),
        'alpha': hp.choice('alpha', [0.7]),
        'rest': hp.choice('rest', [False]),
    },
    'DeepFM': {
        'lr': hp.choice('lr', [0.0002]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [50]),
        'rho': hp.choice('rho', [0.0001]),
        'p_lr': hp.choice('p_lr', [0.01]),
        'p_lr_decay': hp.choice('p_lr_decay', [1.0]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.1]),
        'rest': hp.choice('rest', [True]),
    },

    'FM': {
        'lr': hp.choice('lr', [0.001]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'amsgrad': hp.choice('amsgrad', [True]),
        'batch_n_ID': hp.choice('batch_n_ID', [100]),
        'rho': hp.choice('rho', [1e-09]),
        'p_lr': hp.choice('p_lr', [0.001]),
        'p_lr_decay': hp.choice('p_lr_decay', [0.001]),
        'gamma': hp.choice('gamma', [0.5]),
        'alpha': hp.choice('alpha', [0.5]),
        'rest': hp.choice('rest', [True]),
    },
}

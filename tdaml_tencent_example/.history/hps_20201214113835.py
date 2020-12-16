from hyperopt import hp

models = {
    'FM': {
        'lr': hp.choice('lr', [0.001]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [30]),
        'rho': hp.choice('rho', [0.0001]),
        'p_lr': hp.choice('p_lr', [0.001]),
        'p_lr_decay': hp.choice('p_lr_decay', [0.1]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.1]),
        'rest': hp.choice('rest', [True]),
    },

    'DNN': {
        'lr': hp.choice('lr', [0.001]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [20]),
        'rho': hp.choice('rho', [1e-5]),
        'p_lr': hp.choice('p_lr', [0.0001]),
        'p_lr_decay': hp.choice('p_lr_decay', [0.001]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.1]),
        'rest': hp.choice('rest', [False]),
    },
    'WD': {
        'lr': hp.choice('lr', [0.01]),
        'weight_decay': hp.choice('weight_decay', [1e-5]),
        'amsgrad': hp.choice('amsgrad', [True]),
        'batch_n_ID': hp.choice('batch_n_ID', [5]),
        'rho': hp.choice('rho', [0.01]),
        'p_lr': hp.choice('p_lr', [0.001]),
        'p_lr_decay': hp.choice('p_lr_decay', [0.1]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.1]),
        'rest': hp.choice('rest', [True]),
    },

    'DeepFM': {
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

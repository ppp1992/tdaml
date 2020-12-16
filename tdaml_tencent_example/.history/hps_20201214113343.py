from hyperopt import hp

models = {
    'FM': {
        'lr': hp.choice('lr', [1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 1e-4, 9e-5, 1e-5]),
        'weight_decay': hp.choice('weight_decay', [0, 1e-8, 1e-7, 1e-6, 1e-5]),
        'amsgrad': hp.choice('amsgrad', [False, True]),
        'batch_n_ID': hp.choice('batch_n_ID', [5, 10, 15, 20, 25, 30, 50, 90]),
        'rho': hp.choice('rho', [1e-7, 1e-8, 1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
        'p_lr': hp.choice('p_lr', [0.0001, 0.001, 0.0005, 0.005, 0.01]),
        'p_lr_decay': hp.choice('p_lr_decay', [1.0, 0.1, 0.01, 0.001]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.1]),
        'rest': hp.choice('rest', [False, True]),
    },

    'DNN': {
        'lr': hp.choice('lr', [0.001]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [20]),
        'rho': hp.choice('rho', [1e-6]),
        'p_lr': hp.choice('p_lr', [0.0001]),
        'p_lr_decay': hp.choice('p_lr_decay', [1.0]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.1]),
        'rest': hp.choice('rest', [True]),
    },
    'WD': {
        'lr': hp.choice('lr', [0.01]),
        'weight_decay': hp.choice('weight_decay', [1e-5]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [25]),
        'rho': hp.choice('rho', [0.001]),
        'p_lr': hp.choice('p_lr', [0.001]),
        'p_lr_decay': hp.choice('p_lr_decay', [1.0]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.5]),
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

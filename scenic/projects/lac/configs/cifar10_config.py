# pylint: disable=line-too-long

import ml_collections

_CIFAR10_TRAIN_SIZE = 60_000
VARIANT = 'B/16'
MODE = 'E'  # B -> Per Batch, E -> Per Example, T -> Per Token


def get_config(runlocal=''):
    """Returns the LAC UVit experiment configuration for Cifar10."""

    runlocal = bool(runlocal)

    config = ml_collections.ConfigDict()
    config.experiment_name = 'lac-vit'
    
    # Dataset.
    config.dataset_name = 'cifar10'
    config.data_dtype_str = 'float32'
    config.dataset_configs = ml_collections.ConfigDict()
    config.count_flops = False

    # Model.
    version, patch = VARIANT.split('/')
    config.model_name = 'lac_vit_multilabel_classification'
    config.model = ml_collections.ConfigDict()
    config.model.hidden_size = {'Ti': 192,
                                'S': 384,
                                'B': 768,
                                'L': 1024,
                                'H': 1280}[version]
    config.model.patches = ml_collections.ConfigDict()
    config.model.patches.size = [int(patch), int(patch)]
    config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
    config.model.mlp_dim = {'Ti': 768,
                            'S': 1536,
                            'B': 3072,
                            'L': 4096,
                            'H': 5120}[version]
    config.model.num_layers = {'Ti': 12,
                               'S': 12,
                               'B': 12,
                               'L': 24,
                               'H': 32}[version]
    
    # Representation Size has to be set to the hidden size, to match
    config.model.representation_size = {'Ti': 192,
                                        'S': 384,
                                        'B': 768,
                                        'L': 1024,
                                        'H': 1280}[version]
    config.model.classifier = 'token'
    config.model.attention_dropout_rate = 0.
    config.model.dropout_rate = 0.1
    config.model_dtype_str = 'float32'
    config.model.parameter_sharing = False

    # Lac config
    config.model.lac_config = ml_collections.ConfigDict()
    config.model.lac_config.vit = False
    config.model.lac_config.train_alpha = 0.8
    config.model.lac_config.test_alpha = 0.8
    config.model.lac_config.use_mask = False
    config.model.lac_config.mode = MODE
    config.model.lac_config.axis = {
        'B': None,
        'E': (1, 2),
        'T': -1
    }[MODE]
    config.model.lac_config.state_slice = {
        'B': slice(0),
        'E': slice(0, 1),
        'T': slice(0, 2)
    }[MODE]
    
    # Training.
    config.trainer_name = 'classification_trainer'
    config.optimizer = 'adamw'
    config.optimizer_configs = ml_collections.ConfigDict()
    config.optimizer_configs.beta1 = 0.9
    config.optimizer_configs.beta2 = 0.999
    config.optimizer_configs.weight_decay = 0.3
    config.explicit_weight_decay = None  # No explicit weight decay
    config.l2_decay_factor = None
    config.max_grad_norm = 1.0
    config.label_smoothing = None
    config.num_training_epochs = 300
    config.log_eval_steps = 1000
    config.batch_size = 1024
    config.rng_seed = 42
    config.init_head_bias = -10.0

    # Learning rate.
    steps_per_epoch = _CIFAR10_TRAIN_SIZE // config.batch_size
    total_steps = config.num_training_epochs * steps_per_epoch
    base_lr = 3e-3
    config.lr_configs = ml_collections.ConfigDict()
    config.lr_configs.learning_rate_schedule = 'compound'
    config.lr_configs.factors = 'constant*linear_warmup*cosine_decay'
    config.lr_configs.total_steps = total_steps
    config.lr_configs.steps_per_cycle = total_steps
    config.lr_configs.end_learning_rate = 1e-5
    config.lr_configs.warmup_steps = 10_000
    config.lr_configs.base_learning_rate = base_lr

    # Logging.
    config.write_summary = True
    config.xprof = True  # Profile using xprof.
    config.checkpoint = True  # Do checkpointing.
    config.checkpoint_steps = 5000
    config.debug_train = False  # Debug mode during training.
    config.debug_eval = False  # Debug mode during eval.

    return config
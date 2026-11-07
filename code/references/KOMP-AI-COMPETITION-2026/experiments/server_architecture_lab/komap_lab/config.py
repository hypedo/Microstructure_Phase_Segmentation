import copy
import json
from pathlib import Path

DEFAULT = {
    'schema_version': 2, 'id': 'B', 'description': 'R022 recipe: ImageNet V2 ResNet101, Context448, bilinear',
    'model': {'backbone': 'resnet101', 'decoder': 'unet', 'upsampling': 'bilinear', 'hf': False,
              'cross': 'none', 'dilation': 'none', 'weighted_skip': False, 'hybrid': 'none', 'mamba_blocks': 0,
              'initialization': 'imagenet'},
    'normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
    'train': {'epochs': 150, 'seed': 42, 'patch': 448, 'target': 224, 'stride': 112, 'samples_per_epoch': 420,
              'batch_size': 4, 'workers': 0, 'lr': .0003, 'weight_decay': .0001,
              'rare_fraction': .10, 'enriched_probability': .5, 'rare_classes': [1, 2],
              'brightness_contrast_probability': .5, 'gamma_probability': .3, 'noise_probability': .15,
              'precision': 'fp32',
              'deterministic': False},
    'loss': {'ce_weight': .5, 'dice_weight': .25, 'lovasz_weight': .25, 'class_weights': [1, 1, 2, 1]},
    'evaluation': {'batch_size': 4, 'absent': 'one', 'final_d4': True},
    'ssl': {'epochs': 50, 'patch': 224, 'batch_size': 32, 'steps_per_epoch': 70, 'lr': .0003, 'weight_decay': .0001,
            'temperature': .07, 'projection_width': 512, 'embedding_width': 128},
}


def validate(config):
    if set(config) != set(DEFAULT) or config['schema_version'] != 2:
        raise ValueError('Unsupported or incomplete config schema')
    for key in ('model', 'normalization', 'train', 'loss', 'evaluation', 'ssl'):
        if set(config[key]) != set(DEFAULT[key]):
            raise ValueError(f'Unexpected/missing fields in config.{key}')
    m, t, s = config['model'], config['train'], config['ssl']
    for key, options in {'backbone': ('resnet34', 'resnet101'), 'decoder': ('unet', 'unet_rrcu', 'unetpp', 'upernet'),
                         'upsampling': ('dysample', 'bilinear'), 'cross': ('none', 'attention', 'concat'),
                         'dilation': ('none', 'adaptive', 'fixed'), 'hybrid': ('none', 'gated', 'concat'),
                         'initialization': ('imagenet', 'none', 'micronet', 'ssl')}.items():
        if m[key] not in options:
            raise ValueError(f'Invalid model.{key}: {m[key]}')
    if m['decoder'] != 'unet' and (m['dilation'] != 'none' or m['weighted_skip']):
        raise ValueError('Dilation/weighted skip currently require UNet decoder')
    if type(m['mamba_blocks']) is not int or not 0 <= m['mamba_blocks'] <= 4:
        raise ValueError('mamba_blocks must be 0..4')
    for k in ('hf', 'weighted_skip'):
        if type(m[k]) is not bool:
            raise ValueError(f'model.{k} must be boolean')
    for key in ('epochs', 'patch', 'stride', 'samples_per_epoch', 'batch_size'):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f'train.{key} must be positive integer')
    if t['patch'] % 32 or t['target'] % 32 or t['patch'] != 2 * t['target'] or not 0 < t['stride'] <= t['target']:
        raise ValueError('Require Context448 = 2× target224, both divisible by 32; stride <= target')
    if t['batch_size'] < 2 or t['samples_per_epoch'] % t['batch_size']:
        raise ValueError('BN training requires batch >=2 and samples_per_epoch divisible by batch_size')
    if t['workers'] != 0:
        raise ValueError('workers=0 required for reproducible epoch-boundary RNG resume')
    if t['precision'] != 'fp32':
        raise ValueError('Only fp32 supported: keep the original loss/evaluation precision')
    if not 0 <= t['rare_fraction'] <= 1 or not 0 <= t['enriched_probability'] <= 1:
        raise ValueError('Sampling fractions must be in 0..1')
    if len(config['normalization']['mean']) != 3 or len(config['normalization']['std']) != 3 or min(config['normalization']['std']) <= 0:
        raise ValueError('Context448 recipe expects 3-channel ImageNet normalization')
    if [config['loss'][k] for k in ('ce_weight', 'dice_weight', 'lovasz_weight')] != [.5, .25, .25]:
        raise ValueError('R022 loss weights are fixed at CE=.5, Dice=.25, Lovasz=.25')
    if config['loss']['class_weights'] != [1, 1, 2, 1]:
        raise ValueError('R022 class weights must be [1,1,2,1]')
    if t['lr'] <= 0 or t['weight_decay'] < 0:
        raise ValueError('Invalid learning rate, weight decay or normalization')
    if config['evaluation']['absent'] != 'one' or config['evaluation']['batch_size'] < 1:
        raise ValueError('Evaluation uses absent=one and positive batch_size')
    if s['epochs'] <= 0 or s['patch'] <= 0 or s['batch_size'] < 2 or s['steps_per_epoch'] <= 0 or s['temperature'] <= 0:
        raise ValueError('Invalid SSL settings')
    return config


def load_config(path):
    return validate(json.loads(Path(path).read_text(encoding='utf-8')))


def new_config(experiment_id, description, **model_changes):
    config = copy.deepcopy(DEFAULT)
    config.update(id=experiment_id, description=description)
    config['model'].update(model_changes)
    return validate(config)

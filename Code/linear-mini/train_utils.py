# Code was adapted from https://github.com/guy-dar/lra-benchmarks
# Dar, G. (2023). lra-benchmarks. GitHub. https://github.com/guy-dar/lra-benchmarks.

# Code was also inspired from https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/utils/train_utils.py
# Google Research. (2020). long-range-arena/lra_benchmarks/utils/train_utils.py. GitHub. 
# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/utils/train_utils.py.

import numpy as np
from torch.optim.lr_scheduler import LambdaLR

# Create learning rate scheduler. From config, we can obtain the required
#   parameters to adjust the learning rate. The specific operation to adjust
#   learning rate is obtained from factors (which will be a string)
# factors:  string of factors (delimited by *) that define operations to affect 
#           learning rate
# config:                   config for parameters of learning rate scheulder
# config.learning_rate:     starting constant for learning rate scheduler
# config.warmup_steps:      required number of steps for warmup
# config.decay_factor:      decay factor for learning rate
# config.steps_per_decay:   determine frequency for learning rate decay
# config.steps_per_cycle:   steps per cycle when using cosine decay
def create_learning_rate_scheduler(factors, config):
    
    # Extract factors from factors argument
    factors = [n.strip() for n in factors.split('*')]
    
    # Obtain individual factors from config. If not present, use default values
    base_learning_rate = config.learning_rate
    warmup_steps = config.get('warmup_steps', 1000)
    decay_factor = config.get('decay_factor', 0.5)
    steps_per_decay = config.get('steps_per_decay', 20000)
    steps_per_cycle = config.get('steps_per_cycle', 100000)

    # Step function for learning rate. Each factor will adjust required lerning rate
    def step_fn(step):
        ret = 1.0
        for name in factors:
            if name == 'constant':
                ret *= base_learning_rate
            elif name == 'linear_warmup':
                ret *= np.minimum(1.0, step / warmup_steps)
            elif name == 'rsqrt_decay':
                ret /= np.sqrt(np.maximum(step, warmup_steps))
            elif name == 'rsqrt_normalized_decay':
                ret *= np.sqrt(warmup_steps)
                ret /= np.sqrt(np.maximum(step, warmup_steps))
            elif name == 'decay_every':
                ret *= (decay_factor ** (step // steps_per_decay))
            elif name == 'cosine_decay':
                progress = np.maximum(0.0, (step - warmup_steps) / float(steps_per_cycle))
                ret *= np.maximum(0.0, 0.5 * (1.0 + np.cos(np.pi * (progress % 1.0))))
            else:
                raise ValueError('Unknown factor %s.' % name)
        return ret

    return lambda optimizer: LambdaLR(optimizer, step_fn)

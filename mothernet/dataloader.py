import numpy as np
from torch.utils.data import DataLoader

import mothernet.priors as priors
from mothernet.priors import ClassificationAdapterPrior, BagPrior, BooleanConjunctionPrior, RegressionAdapterPrior


class PriorDataLoader(DataLoader):
    def __init__(self, prior, num_steps, batch_size, min_eval_pos, max_eval_pos, n_samples, device, num_features):
        self.prior = prior
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.min_eval_pos = min_eval_pos
        self.max_eval_pos = max_eval_pos
        self.n_samples = n_samples
        self.device = device
        self.num_features = num_features
        self.epoch_count = 0

    def gbm(self, epoch=None):
        # Actually can only sample up to max_eval_pos-1 but that's how it was in the original code
        single_eval_pos = np.random.randint(self.min_eval_pos, self.max_eval_pos)
        batch = self.prior.get_batch(batch_size=self.batch_size, n_samples=self.n_samples, num_features=self.num_features, device=self.device,
                                     epoch=epoch,
                                     single_eval_pos=single_eval_pos)
        # we return sampled hyperparameters from get_batch for testing but we don't want to use them as style.
        x, y, target_y, _ = batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
        return (None, x, y), target_y, single_eval_pos

    def __len__(self):
        return self.num_steps

    def get_test_batch(self):  # does not increase epoch_count
        return self.gbm(epoch=self.epoch_count)

    def __iter__(self):
        self.epoch_count += 1
        return iter(self.gbm(epoch=self.epoch_count - 1) for _ in range(self.num_steps))


def get_dataloader(prior_config, dataloader_config, device):

    prior_type = prior_config['prior_type']
    if 'regression' in prior_type:
        gp_flexible = RegressionAdapterPrior(priors.GPPrior(prior_config['gp']), **prior_config['regression'])
        mlp_flexible = RegressionAdapterPrior(priors.MLPPrior(prior_config['mlp']), **prior_config['regression'])
    else:
        gp_flexible = ClassificationAdapterPrior(priors.GPPrior(prior_config['gp']), **prior_config['classification'])
        mlp_flexible = ClassificationAdapterPrior(priors.MLPPrior(prior_config['mlp']), **prior_config['classification'])

    if 'prior_bag' in prior_type:
        # Prior bag combines priors
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                         prior_weights={'mlp': 0.961, 'gp': 0.039})
    elif "boolean_only" in prior_type:
        prior = BooleanConjunctionPrior(hyperparameters=prior_config['boolean'])
    elif "bag_boolean" in prior_type:
        boolean = BooleanConjunctionPrior(hyperparameters=prior_config['boolean'])
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible, 'boolean': boolean},
                         prior_weights={'mlp': 0.9, 'gp': 0.02, 'boolean': 0.08})
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")

    return PriorDataLoader(prior=prior, num_steps=dataloader_config['num_steps'], batch_size=dataloader_config['batch_size'],
                           n_samples=prior_config['n_samples'], min_eval_pos=dataloader_config['min_eval_pos'],
                           max_eval_pos=dataloader_config['max_eval_pos'], device=device,
                           num_features=prior_config['num_features'])

import torch
import torch.autograd as autograd

from wilds.common.utils import split_into_groups


class GroupDRO:
    def __init__(self, n_groups, device="cpu"):
        self.n_groups = n_groups
        self._device = device
        self.group_weights_step_size = 0.01
        self.group_weights = torch.ones(n_groups).to(device) / n_groups

    def set_device(self, device):
        self.group_weights = self.group_weights.to(device)
        self._device = device

    def reweighted_loss(self, groups, losses):
        group_losses = torch.zeros(self.n_groups).to(self._device)
        for i in range(self.n_groups):
            if any(groups == i):
                group_losses[i] = torch.mean(losses[groups == i])
        loss = self.group_weights @ group_losses
        return loss, group_losses.detach()

    def update_group_weights(self, group_losses):
        self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*group_losses)
        self.group_weights = self.group_weights/self.group_weights.sum()


# Taken from: https://github.com/p-lambda/wilds
def coral_penalty(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0.
    
    if x.dim() > 2:
        # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
        # we flatten to Tensors of size (*, feature dimensionality)
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))

    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff


# Taken from: https://github.com/p-lambda/wilds
def coral_loss(groups, features, device):
    unique_groups, group_indices, _ = split_into_groups(groups)
    n_groups_per_batch = unique_groups.numel()

    # Compute penalty - perform pairwise comparisons between features of all the groups
    penalty = torch.zeros(1, device=device)
    for i_group in range(n_groups_per_batch):
        for j_group in range(i_group+1, n_groups_per_batch):
            penalty += coral_penalty(features[group_indices[i_group]], features[group_indices[j_group]])
    if n_groups_per_batch > 1:
        penalty /= (n_groups_per_batch * (n_groups_per_batch-1) / 2) # get the mean penalty
    return penalty



# Adapted from: https://github.com/p-lambda/wilds/blob/main/examples/algorithms/IRM.py
def irm_penalty(groups, logits, y, loss_fn, device):
    """Group-wise gradient norm penalty for Invariant Risk Minimization (IRM)."""
    _, group_indices, _ = split_into_groups(groups)
    penalty = torch.zeros(1, device=device)
    scale = torch.tensor(1.).to(device).requires_grad_()

    for i_group in group_indices:
        group_losses = loss_fn(scale * logits[i_group], y[i_group])
        grad_1 = autograd.grad(group_losses[0::2].mean(), [scale], create_graph=True)[0]
        grad_2 = autograd.grad(group_losses[1::2].mean(), [scale], create_graph=True)[0]
        penalty += torch.sum(grad_1 * grad_2)
    return penalty
import torch


def accuracy(predictions, targets):
    """
    Calculates the top-1 accuracy between predictions and targets

    @param predictions (torch.Tensor): Tensor of shape [batch_size, N]
        containing scores/probabilites for each class n in N
    @param targets (torch.Tensor): Tensor of shape [batch_size] containg the correct
        classes
    """
    return torch.eq(predictions.topk(1)[1].view(-1),
                    targets.view(-1)).sum().item() / targets.size(0)

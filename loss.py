import torch
import torch.nn as nn
import torch.nn.functional as F


class FZLPRLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, sample_weight=None):
        targets = targets.float()

        losses = []
        valid_weights = []

        for i in range(logits.size(0)):
            logit = logits[i]
            target = targets[i]

            pos_logits = logit[target == 1]
            neg_logits = logit[target == 0]

            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue

            diff = neg_logits.unsqueeze(1) - pos_logits.unsqueeze(0)

            base_loss = F.softplus(diff)
            focal_weight = torch.sigmoid(diff).pow(self.gamma)

            loss = (focal_weight * base_loss).mean()
            losses.append(loss)

            if sample_weight is not None:
                valid_weights.append(sample_weight[i])

        if len(losses) == 0:
            return logits.sum() * 0.0

        losses = torch.stack(losses)

        if sample_weight is not None:
            valid_weights = torch.stack(valid_weights).to(logits.device)
            losses = losses * valid_weights

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class BCEFZLPRLoss(nn.Module):
    def __init__(
        self,
        pos_weight=None,
        alpha=0.7,
        gamma=2.0
    ):
        super().__init__()

        self.alpha = alpha

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="none"
        )

        self.fzlpr = FZLPRLoss(
            gamma=gamma,
            reduction="mean"
        )

    def forward(self, logits, targets, sample_weight=None):
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        bce_loss = bce_loss.mean(dim=1)

        if sample_weight is not None:
            sample_weight = sample_weight.to(logits.device)
            bce_loss = bce_loss * sample_weight

        bce_loss = bce_loss.mean()

        rank_loss = self.fzlpr(
            logits,
            targets,
            sample_weight=sample_weight
        )

        loss = self.alpha * bce_loss + (1.0 - self.alpha) * rank_loss

        return loss
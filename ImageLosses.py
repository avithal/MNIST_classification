import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageLosses:
    # 1️⃣ Classification Losses
    def cross_entropy(self, pred, target):
        return F.cross_entropy(pred, target)

    def focal_loss(self, pred, target, alpha=1, gamma=2):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        return (alpha * (1 - pt) ** gamma * ce_loss).mean()

    def focal_loss_simple(self,inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

    def label_smoothing(self, pred, target, smoothing=0.1):
        n_class = pred.size(1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing / (n_class - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1 - smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))

    # 2Regression / Reconstruction Losses
    def l1_loss(self, pred, target):
        return F.l1_loss(pred, target)

    def l2_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def huber_loss(self, pred, target, delta=1.0):
        return F.smooth_l1_loss(pred, target, beta=delta)

    def perceptual_loss(self, vgg_features_pred, vgg_features_target):
        """Example: L2 between features from a pretrained VGG"""
        return F.mse_loss(vgg_features_pred, vgg_features_target)

    # 3️⃣ Segmentation / Similarity Losses
    def dice_loss(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        num = 2 * (pred * target).sum()
        den = pred.sum() + target.sum() + eps
        return 1 - num / den

    def iou_loss(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        return 1 - (inter + eps) / (union + eps)

    def bce_dice_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target) + self.dice_loss(pred, target)

    # 4️⃣ Adversarial & Contrastive Losses
    def gan_discriminator_loss(self, real_pred, fake_pred):
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        return (real_loss + fake_loss) / 2

    def gan_generator_loss(self, fake_pred):
        return F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))

    def contrastive_loss(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(logits, labels)

    # 5️⃣ Self-Supervised / Embedding Losses
    def cosine_similarity_loss(self, z1, z2):
        return 1 - F.cosine_similarity(z1, z2).mean()

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        return F.triplet_margin_loss(anchor, positive, negative, margin=margin)

    def barlow_twins_loss(self, z1, z2, lambd=5e-3):
        N, D = z1.size()
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        c = torch.mm(z1_norm.T, z2_norm) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
        return on_diag + lambd * off_diag


# Example usage
if __name__ == "__main__":
    loss_fn = ImageLosses()
    pred = torch.randn(8, 10, requires_grad=True)
    target = torch.randint(0, 10, (8,))
    print("Cross Entropy Loss:", loss_fn.cross_entropy(pred, target).item())

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask):
        loss = self.base_loss(pred, target)
        return (loss * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1e-6)

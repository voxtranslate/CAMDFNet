import torch
import torch.nn as nn


# =============================================
# LOSS FUNCTIONS
# =============================================

class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def _prepare_masks_for_loss(mask_pred: torch.Tensor, mask_true, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(mask_true, torch.Tensor):
        if mask_true.shape[-2:] != mask_pred.shape[-2:]:
            mask_true = F.interpolate(mask_true.float(), size=mask_pred.shape[-2:], mode='nearest')
        return mask_pred, mask_true.to(device)
    elif isinstance(mask_true, list):
        resized_masks = []
        for m in mask_true:
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m).float()
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.dim() == 3:
                m = m.unsqueeze(0)
            m_resized = F.interpolate(m.float(), size=mask_pred.shape[-2:], mode='nearest')
            resized_masks.append(m_resized.squeeze(0))
        mask_true_stacked = torch.stack(resized_masks).to(device)
        return mask_pred, mask_true_stacked
    else:
        raise TypeError(f"Unexpected mask_true type: {type(mask_true)}")

class CAMDFNetLoss(nn.Module):
    def __init__(self, config: LossConfig, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(config.focal_alpha, config.focal_gamma) if config.use_focal else None
        self.class_weights = class_weights
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Tuple) -> Dict[str, torch.Tensor]:
        images, mask_true, labels = targets[:3]
        logits = outputs['logits']
        mask_pred = outputs['mask']
        uncertainty = outputs['uncertainty']
        
        device = logits.device
        
        mask_pred_for_loss, mask_true_for_loss = _prepare_masks_for_loss(mask_pred, mask_true, device)
        
        if self.config.use_focal and self.focal_loss:
            cls_loss = self.focal_loss(logits, labels)
        else:
            cls_loss = F.cross_entropy(
                logits, labels,
                weight=self.class_weights,
                label_smoothing=self.config.label_smoothing
            )
        
        dice_loss = self.dice_loss(mask_pred_for_loss, mask_true_for_loss)
        bce_loss = F.binary_cross_entropy_with_logits(mask_pred_for_loss, mask_true_for_loss)
        mask_reg = (torch.sigmoid(mask_pred_for_loss).mean() - 0.5).abs()
        aux_loss = torch.tensor(0.0, device=device)
        
        total_loss = (
            self.config.lambda_cls * cls_loss +
            self.config.lambda_dice * dice_loss +
            self.config.lambda_bce * bce_loss +
            self.config.lambda_mask_reg * mask_reg +
            self.config.lambda_aux * aux_loss
        )
        
        return {
            'total': total_loss,
            'classification': cls_loss,
            'dice': dice_loss,
            'bce': bce_loss,
            'mask_reg': mask_reg,
            'auxiliary': aux_loss
        }
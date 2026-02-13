import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torchvision import models, transforms
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any

# =============================================
# MODULAR MODEL COMPONENTS
# =============================================
class SpectralPreProcessor(nn.Module):
    def __init__(self, in_channels: int = 3, freq_ratios: List[float] = [0.2, 0.5, 1.0]):
        super().__init__()
        self.ratios = freq_ratios
        self.filters = nn.ParameterList([
            nn.Parameter(torch.randn(1, in_channels, 1, 1, dtype=torch.complex64))
            for _ in freq_ratios
        ])
        for f in self.filters:
            nn.init.normal_(f.real, mean=1.0, std=0.01)
            nn.init.normal_(f.imag, mean=0.0, std=0.01)
        self.activations = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        out_spatial = torch.zeros_like(x)
        for i, ratio in enumerate(self.ratios):
            h_lim = max(1, int(x_fft.shape[2] * ratio))
            w_lim = max(1, int(x_fft.shape[3] * ratio))
            mask = torch.zeros_like(x_fft)
            mask[:, :, :h_lim, :w_lim] = x_fft[:, :, :h_lim, :w_lim]
            out_spatial += torch.fft.irfft2(mask * self.filters[i], s=(h, w), norm='ortho')
        
        self.activations['spectral_output'] = out_spatial.detach()
        return out_spatial / len(self.ratios)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, pool_sizes: List[int] = [1, 2, 3, 6]):
        super().__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        self.activations = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pyramids = [x]
        for i, stage in enumerate(self.stages):
            pooled = stage(x)
            pyramids.append(F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True))
            self.activations[f'pool_stage_{i}'] = pooled.detach()
        
        output = torch.cat(pyramids, dim=1)
        self.activations['pyramid_output'] = output.detach()
        return output

class AttentionRefinementModule(nn.Module):
    """Refine segmentation masks using attention"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, 1)
        self.activations = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.bn1(self.conv1(x)))
        attention = torch.sigmoid(self.conv2(x1))
        
        self.activations['attention_input'] = x.detach()
        self.activations['attention_output'] = attention.detach()
        
        return attention

class CAMDFNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.activations = {}
        self.gradients = {}
        is_large = "large" in config.name.lower()
        
        if is_large:
            self.vit = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
            self.hidden_dim, self.num_heads = 1024, 16
            eff = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
            cnn_ch = 224
        else:
            self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.hidden_dim, self.num_heads = 768, 12
            eff = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            cnn_ch = 160
        
        self.vit_layers = self.vit.encoder.layers
        self.spectral_pre = SpectralPreProcessor()
        self.cnn_backbone = nn.Sequential(*list(eff.features.children())[:6])
        self.proj2 = nn.Conv2d(cnn_ch, self.hidden_dim, 1)
        self.pyramid_pool = PyramidPoolingModule(self.hidden_dim)
        
        self.attention_refine = AttentionRefinementModule(self.hidden_dim * 2)
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        
        self.cross_attention = nn.MultiheadAttention(self.hidden_dim, num_heads=self.num_heads, batch_first=False)
        
        # Classifier - IMPORTANT: num_classes is set automatically from dataset
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.hidden_dim, config.num_classes)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.hidden_dim, 1), nn.Sigmoid()
        )
        
        # Register hooks for all intermediate layers
        self._register_all_hooks()
    
    def _register_all_hooks(self):
        def save_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    if len(output) == 2:
                        if output[0] is not None and hasattr(output[0], 'detach'):
                            self.activations[name + '_output'] = output[0].detach()
                        if output[1] is not None and hasattr(output[1], 'detach'):
                            self.activations[name + '_weights'] = output[1].detach()
                    else:
                        for i, out in enumerate(output):
                            if out is not None and hasattr(out, 'detach'):
                                self.activations[f'{name}_tuple_{i}'] = out.detach()
                else:
                    if output is not None and hasattr(output, 'detach'):
                        self.activations[name] = output.detach()
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                if isinstance(grad_output, tuple):
                    if len(grad_output) > 0 and grad_output[0] is not None:
                        self.gradients[name] = grad_output[0].detach()
                elif grad_output is not None and hasattr(grad_output, 'detach'):
                    self.gradients[name] = grad_output.detach()
            return hook
        
        self.spectral_pre.register_forward_hook(save_activation('spectral_pre'))
        
        for i, layer in enumerate(self.cnn_backbone):
            layer.register_forward_hook(save_activation(f'cnn_layer_{i}'))
        
        self.proj2.register_forward_hook(save_activation('proj2'))
        self.pyramid_pool.register_forward_hook(save_activation('pyramid_pool'))
        self.attention_refine.register_forward_hook(save_activation('attention_refine'))
        
        for i, layer in enumerate(self.segmentation_head):
            layer.register_forward_hook(save_activation(f'seg_layer_{i}'))
        
        self.cross_attention.register_forward_hook(save_activation('cross_attention'))
        self.classifier.register_forward_hook(save_activation('classifier'))
        self.uncertainty_head.register_forward_hook(save_activation('uncertainty_head'))
    
    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        
        self.activations.clear()
        self.activations['input'] = x.detach()
        
        x_res = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        x_spec = self.spectral_pre(x_res)
        
        v = self.vit
        x_v = v._process_input(x_spec)
        n = x_v.shape[0]
        cls_token = v.class_token.expand(n, -1, -1)
        x_v = torch.cat((cls_token, x_v), dim=1)
        x_v = v.encoder(x_v)
        
        vit_spatial = x_v[:, 1:].transpose(1, 2).reshape(b, self.hidden_dim, 14, 14)
        self.activations['vit_spatial'] = vit_spatial.detach()
        self.activations['vit_sequence'] = x_v.detach()
        
        # Get pyramid pooled features
        pooled_features = self.pyramid_pool(vit_spatial)
        
        # Get attention weights for mask refinement
        attention_weights = self.attention_refine(pooled_features)
        
        # Generate mask with attention refinement
        mask_logits = self.segmentation_head(pooled_features)
        attention_weights_upsampled = F.interpolate(attention_weights, size=(224, 224), mode='bilinear', align_corners=True)
        refined_mask = mask_logits * attention_weights_upsampled
        self.activations['mask_logits'] = mask_logits.detach()
        self.activations['refined_mask'] = refined_mask.detach()
        
        gated_x = x_spec * (0.5 + 0.5 * torch.sigmoid(refined_mask))
        self.activations['gated_x'] = gated_x.detach()
        
        cnn_features = self.cnn_backbone(gated_x)
        self.activations['cnn_features_raw'] = cnn_features.detach()
        
        cnn_aligned = F.interpolate(self.proj2(cnn_features), size=(14, 14), mode='bilinear', align_corners=True)
        self.activations['cnn_aligned'] = cnn_aligned.detach()
        
        cnn_seq = cnn_aligned.flatten(2).permute(2, 0, 1)
        vit_seq = x_v.permute(1, 0, 2)
        
        fused_seq, cross_weights = self.cross_attention(cnn_seq, vit_seq, vit_seq, need_weights=True)
        
        fused_feat = fused_seq.permute(1, 2, 0).reshape(b, self.hidden_dim, 14, 14)
        self.activations['fused_features'] = fused_feat.detach()
        
        cls_logits = self.classifier(fused_feat)
        uncertainty = self.uncertainty_head(fused_feat)
        mask_upscaled = F.interpolate(refined_mask, size=(h, w), mode='bilinear', align_corners=True)
        
        self.activations['cls_logits'] = cls_logits.detach()
        self.activations['uncertainty'] = uncertainty.detach()
        self.activations['final_mask'] = mask_upscaled.detach()
        
        return {
            'logits': cls_logits,
            'mask': mask_upscaled,
            'uncertainty': uncertainty,
            'cross_attention_weights': cross_weights,
            'spectral_image': x_spec,
            'vit_features': vit_spatial,
            'cnn_features': cnn_aligned,
            'fused_features': fused_feat,
            'vit_sequence': x_v,
            'attention_weights': attention_weights
        }

    @torch.no_grad()
    def get_vit_self_attention(self, x: torch.Tensor, 
                               layer_indices: List[int] = None,
                               return_all_heads: bool = False) -> Dict[str, Any]:
        self.eval()
        b, _, h, w = x.shape
        
        x_res = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        x_spec = self.spectral_pre(x_res)
        v = self.vit
        
        x_v = v._process_input(x_spec)
        n = x_v.shape[0]
        cls_token = v.class_token.expand(n, -1, -1)
        x_v = torch.cat((cls_token, x_v), dim=1)
        x_v = x_v + v.encoder.pos_embedding
        
        if layer_indices is None:
            layer_indices = list(range(len(v.encoder.layers)))
        
        attention_maps = {}
        
        current_x = x_v
        for idx, layer in enumerate(v.encoder.layers):
            if idx in layer_indices:
                norm_x = layer.ln_1(current_x)
                qkv = F.linear(norm_x, layer.self_attention.in_proj_weight, 
                              layer.self_attention.in_proj_bias)
                
                head_dim = self.hidden_dim // self.num_heads
                qkv = qkv.reshape(b, -1, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, v_ = qkv[0], qkv[1], qkv[2]
                
                attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
                attn = attn.softmax(dim=-1)
                
                if return_all_heads:
                    head_attentions = {}
                    for head_idx in range(self.num_heads):
                        cls_attn_head = attn[:, head_idx, 0, 1:]
                        cls_attn_map = cls_attn_head.reshape(b, 14, 14)
                        
                        map_min = cls_attn_map.amin(dim=(1, 2), keepdim=True)
                        map_max = cls_attn_map.amax(dim=(1, 2), keepdim=True)
                        cls_attn_map = (cls_attn_map - map_min) / (map_max - map_min + 1e-8)
                        
                        head_attentions[f'head_{head_idx}'] = cls_attn_map
                    
                    attention_maps[f'layer_{idx}'] = head_attentions
                else:
                    attn_avg = attn.mean(dim=1)
                    cls_attn = attn_avg[:, 0, 1:]
                    cls_attn_map = cls_attn.reshape(b, 14, 14)
                    
                    map_min = cls_attn_map.amin(dim=(1, 2), keepdim=True)
                    map_max = cls_attn_map.amax(dim=(1, 2), keepdim=True)
                    cls_attn_map = (cls_attn_map - map_min) / (map_max - map_min + 1e-8)
                    
                    attention_maps[f'layer_{idx}'] = cls_attn_map
            
            current_x = layer(current_x)
        
        return attention_maps
    
    def get_all_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations.copy()
    
    def get_all_gradients(self) -> Dict[str, torch.Tensor]:
        return self.gradients.copy()
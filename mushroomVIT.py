import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class PatchEmbedding(nn.Module):
  def __init__(self, 
               img_size:   int,
               patch_size: int,
               in_ch:      int,
               embed_dim:  int):
    super().__init__()
    assert img_size % patch_size == 0, f"img_size {img_size} must be divisible by patch size ({patch_size})"
    #cut an image into patches and apply a linear proj
    #a single convolution with kernel=stride=patch_size accomplishes this
    self.num_patches = (img_size // patch_size) ** 2
    self.proj = nn.Conv2d(in_ch, 
                          embed_dim,
                          kernel_size = patch_size,
                          stride=patch_size)
  
  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
    #X is batch x ch x height x width
    #WANT: batch x N (patches) x D (embed_dim)
    x = self.proj(x) #CNN into B x D x gh x gw
    x = x.flatten(2) #flatten into B x D x N
    x = x.transpose(1,2) #transpose into B x N x D
    return x


###############################################################
###############################################################

class TransformerEncoderBlock(nn.Module):
  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      mlp_ratio: float = 4.0,

      attn_dropout: float = 0.0,
      mlp_dropout: float= 0.0
  ):
    
    super().__init__()
    self.norm1 = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(
      embed_dim,
      num_heads,
      dropout=attn_dropout,
      batch_first=True
    )
    self.norm2 = nn.LayerNorm(embed_dim)
    mlp_hidden = int(embed_dim * mlp_ratio)
    self.mlp = nn.Sequential(
      nn.Linear(embed_dim, mlp_hidden),
      nn.GELU(), #gaussian cdf inspired regularization 
      nn.Dropout(mlp_dropout),
      nn.Linear(mlp_hidden, embed_dim),
      nn.Dropout(mlp_dropout),
    )

  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
    #self-attention with pre-norm and residual
    normed = self.norm1(x) 
    attn_out, _ = self.attn(normed, normed, normed)
    x = x + attn_out

    #MLP with pre-norm and residual
    x = x + self.mlp(self.norm2(x))
    return x
  


###############################################################
###############################################################

class MushroomVIT(nn.Module):
  def __init__(
      self,
      img_size:     int   = 224,
      patch_size:   int   = 16,
      in_ch:        int   = 3,
      embed_dim:    int   = 256,
      num_heads:    int   = 8,
      depth:        int   = 6,
      mlp_ratio:    float = 4.0,
      attn_dropout: float = 0.0,
      mlp_dropout:  float = 0.1,
      head_dropout: float = 0.5,
      num_classes:  int   = 2,
  ):
    super().__init__()
    self.hyperparams = dict(
      img_size=img_size,
      patch_size=patch_size,
      in_ch=in_ch,
      embed_dim=embed_dim,
      num_heads=num_heads,
      depth=depth,
      mlp_ratio=mlp_ratio,
      attn_dropout=attn_dropout,
      mlp_dropout=mlp_dropout,
      head_dropout=head_dropout,
      num_classes=num_classes
    )

    #do patch embedding
    self.patch_embed = PatchEmbedding(img_size,
                                      patch_size,
                                      in_ch,
                                      embed_dim)
    num_patches = self.patch_embed.num_patches

    #learnable tokens + positional embeds + dropout

    self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
    self.pos_embed = nn.Parameter(
                      torch.zeros(1, num_patches + 1, embed_dim)
    )
    self.pos_drop = nn.Dropout(mlp_dropout)

    #actual transformer blocks (all multihead + layer normalizations get supported)
    self.blocks = nn.Sequential(*[
      TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        attn_dropout=attn_dropout,
        mlp_dropout=mlp_dropout,
      ) for _ in range(depth)
    ])

    #classification head
    self.norm = nn.LayerNorm(embed_dim)
    self.head = nn.Sequential(
      nn.Dropout(head_dropout),
      nn.Linear(embed_dim, num_classes)
    )

    self._init_weights()

  def _init_weights(self):
    #init transformer weights
    nn.init.trunc_normal_(self.pos_embed, std=0.02) #init with gaussian but delete any that are two stds out of the mean
    nn.init.trunc_normal_(self.cls_token, std=0.02)
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
  
  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
      B = x.size(0)

      #embed to get B x N (number of patches) x D (embed dim)
      x = self.patch_embed(x)
      cls = self.cls_token.expand(B,  -1, -1) # B x N+1 x D
      x = torch.cat([cls, x], dim=1)

      x = self.pos_drop(x + self.pos_embed)

      #transformer time
      x = self.blocks(x)

      #take CLS token and apply norm + classify
      x = self.norm(x[:,0]) #B x D (cls token only)
      return self.head(x) #B x num classes (2 in this case)


  



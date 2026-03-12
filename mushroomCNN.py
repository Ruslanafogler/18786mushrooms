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







############################################################################
############################################################################

class MushroomCNN(nn.Module):
  '''
  CNN fit for 227x227 pixel wide eRGB mushroom photos
  
  '''

  def __init__(self, 
               num_conv_blocks: int = 3, #num of conv stages
               base_filters:    int = 32, #filters first block, doubles each block
               kernel_size:     int = 3, #kernel filter size (same for all blocks)
               stride:          int = 1, #conv stride
               padding:         int = 1, #conv padding (keep spatial size for stride dim)
               pool_every:      int = 1, #AFTER EVERY pool_every blocks, add a dropout (0=never)
               pool_kernel:     int = 2, # maxpool kernel / stride
               conv_dropout:    float = 0.0, #dropout after each conv block
               depthwise:       bool = False, #depthwise separable convs, cheaper with number of params but not necessarily better?
               fc_hidden:       int = 256, #hidden neurons
               fc_dropout:      float = 0.5, #dropout before finall
               num_classes:     int = 2, #poison or edible
               input_channels:  int = 3, #rgb
               ):

    super().__init__()
    self.hyperparams = dict(
                num_conv_blocks = num_conv_blocks,
                base_filters = base_filters,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                pool_every = pool_every,
                pool_kernel = pool_kernel,
                conv_dropout = conv_dropout,
                depthwise = depthwise,
                fc_hidden = fc_hidden,
                fc_dropout = fc_dropout,
                num_classes = num_classes,
                input_channels = input_channels)
    
    layers = []
    in_ch = input_channels

    for i in range(num_conv_blocks):
      out_ch = base_filters * (2 ** i) #double ch size each block

      if depthwise: #cheaper than regular, not necessarily better tho
        #1 filter per input channel
        #THEN pointwise (1x1 convs)
        if out_ch < in_ch:
           out_ch = in_ch
        
        #for K conv blocks, add a layer to do depth then point:
        layers += [
          nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                    padding=padding, groups=in_ch, bias=False), #depth
          nn.Conv2d(in_ch, out_ch, 1, bias=False) #1x pointwise convs
        ]
      else:
        layers.append(
          nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                    padding=padding, bias=False), #depth
        )
      
      layers += [
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
      ]

      if pool_every > 0 and (i + 1) % pool_every == 0:
        layers.append(nn.MaxPool2d(pool_kernel, pool_kernel)) #insert a max pool every few layers
      
      if conv_dropout > 0.0:
        layers.append(nn.Dropout2d(conv_dropout)) #add a dropout to be triggered based on some prb
      in_ch = out_ch

    ##############################################################################
      
    self.features = nn.Sequential(*layers) #FEATURES is where the CNN is stored
    self.global_pool = nn.AdaptiveAvgPool2d((7, 7)) #

    #flatten and then run through MLP classifier --> 2 classes (poison or edible)
    flat_dim = in_ch * 7 * 7
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(flat_dim, fc_hidden),
      nn.ReLU(inplace=True),
      nn.Dropout(fc_dropout),
      nn.Linear(fc_hidden, num_classes)
    )

  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.global_pool(x) #GLOBAL max pool
    return self.classifier(x) 


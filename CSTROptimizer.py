# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
import torch.nn as nn

class CSTROptimizer(torch.nn.Module):
    """
    MLP model for the CSTR inverse model
    """

    def __init__(self):
        super(CSTROptimizer, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        return y_pred
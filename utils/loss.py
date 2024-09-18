# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Energy_Based_LogMarginal_Loss(nn.Module):

    def __init__(self,):
        super().__init__()
                
    def forward(self, LogSumExpY_real_data, LogSumExpY_sampled_data):

        loss = -1 * (LogSumExpY_real_data - LogSumExpY_sampled_data)
        return torch.mean(loss)



class Reconstruction_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, label, model_output):

        loss = torch.mean(
                    torch.sum(
                        torch.square(label - model_output) , dim = (1,2,3)
                    )
                )
        return loss


class Similarity_Triplet_Loss(nn.Module):

    def __init__(self, margin = 10):
        super().__init__()
        self.margin = margin
        

    def forward(self, anchor_features, positive_features, negative_features):

        positive_distance = torch.sum(
                                torch.square(anchor_features - positive_features), dim = (1,2,3)
                            )
    
        nagative_distance = torch.sum(
                                torch.square(anchor_features - negative_features), dim = (1,2,3)
                            ) 
    
        # triple loss
        loss = torch.mean(
                   torch.relu(positive_distance + self.margin - nagative_distance)
                )

        return loss
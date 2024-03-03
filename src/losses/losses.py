import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


"""
T = tf.keras.backend.flatten(y_true)
    P = tf.keras.backend.flatten(y_pred)
    TP = tf.keras.backend.sum(T * P)
    FN = tf.keras.backend.sum(T * (1 - P))
    FP = tf.keras.backend.sum((1 - T) * P)
    return (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
"""
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.75, smooth=1e-6):
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = 1-alpha
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        T = y_true.flatten()
        P = y_pred.flatten()
        TP = torch.sum(T * P)
        FN = torch.sum(T * (1 - P))
        FP = torch.sum((1 - T) * P)
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        return 1-tversky

class MeanIoU(nn.Module):
    def __init__(self, n_classes=13, smooth=1e-5):
        super(MeanIoU, self).__init__()
        self.smooth = smooth
        self.n_classes = n_classes


    def forward(self, y_pred, y_true):
        y_true = torch.permute(y_true, (0, 2, 3, 1))
        y_pred = torch.permute(y_pred, (0, 2, 3, 1))
        y_true = torch.nn.functional.one_hot(y_true.argmax(3), self.n_classes)
        y_pred = torch.nn.functional.one_hot(y_pred.argmax(3), self.n_classes)
        assert y_true.shape == y_pred.shape, f"{y_true.shape} != {y_pred.shape}" # B, C, H, W
        T = y_true.flatten()
        P = y_pred.flatten()
        TP = torch.sum(T * P)
        FN = torch.sum(T * (1 - P))
        FP = torch.sum((1 - T) * P)
        iou = (TP + self.smooth) / (TP + FN +  FP + self.smooth)

        return iou

class MeanIoUBinary(nn.Module):
    def __init__(self, n_classes=13, smooth=1e-5):
        super(MeanIoUBinary, self).__init__()
        self.smooth = smooth
        self.n_classes = n_classes


    def forward(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape, f"{y_true.shape} != {y_pred.shape}" # B, C, H, W
        T = y_true.flatten()
        P = y_pred.flatten()
        TP = torch.sum(T * P)
        FN = torch.sum(T * (1 - P))
        FP = torch.sum((1 - T) * P)
        iou = (TP + self.smooth) / (TP + FN +  FP + self.smooth)

        return iou

"""
    T = tf.keras.backend.flatten(y_true)
    P = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(T * P)
    dice = (2 * intersection) / (tf.keras.backend.sum(T + P))
"""
class DiceCoeficient(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceCoeficient, self).__init__()
        self.smooth = smooth


    def forward(self, y_pred, y_true):
        T = y_true.flatten()
        P = y_pred.flatten()

        intersection = torch.sum(T * P)
        return ((2 * intersection) + self.smooth) / (T.sum() + P.sum() + self.smooth)




class DiceLoss(nn.Module):
    def __init__(self, beta=1, class_weights=None, class_indexes=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.beta = beta
        self.class_weights = torch.Tensor(class_weights) if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.smooth = smooth

    def forward(self, y_pred, y_true):        
        if self.class_indexes is not None:
            y_true = torch.gather(y_true, 1, torch.Tensor(self.class_indexes))
            y_pred = torch.gather(y_pred, 1, torch.Tensor(self.class_indexes))

        axes = (0, 2, 3)

        # calculate score
        tp = torch.sum(y_true * y_pred, dim=axes)
        fp = torch.sum(y_pred, dim=axes) - tp
        fn = torch.sum(y_true, dim=axes) - tp

        dice = ((1 + self.beta ** 2) * tp + self.smooth) \
                / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        
        if self.class_weights is not None:
            dice = dice * self.class_weights.cuda()

        dice = torch.mean(dice)

        return 1 - dice
    
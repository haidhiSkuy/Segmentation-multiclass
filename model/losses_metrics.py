import torch 
import torch.nn.functional as F

bce = torch.nn.BCEWithLogitsLoss()

def dice_loss(y_pred, y_true, epsilon=1e-6):
    """ 
    y_pred : output of model 
    y_true : one-hot-encoded mask
    """
    y_pred_prob = F.softmax(y_pred, dim=1)
    intersection = torch.sum(y_pred_prob * y_true, dim=(2, 3))
    cardinality_y_pred = torch.sum(y_pred_prob, dim=(2, 3))
    cardinality_y_true = torch.sum(y_true, dim=(2, 3))
    dice_score = (2. * intersection + epsilon) / (cardinality_y_pred + cardinality_y_true + epsilon)
    dice_loss = 1 - dice_score.mean()
    return dice_loss

def criterion(y_pred, y_true): 
    y_true_ohe = F.one_hot(y_true, 5).permute(0,3,1,2)
    y_true_ohe = y_true_ohe.to(torch.float32)
    
    loss = (0.5*bce(y_pred, y_true_ohe)) + (0.5*dice_loss(y_pred, y_true_ohe))
    return loss
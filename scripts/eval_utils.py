import torch

def iou_single_class(predictions, targets, class_id):
    predictions_max = predictions.argmax(1)
    predictions_bool = predictions_max == class_id
    targets_bool = targets == class_id
    union = torch.logical_or(predictions_bool, targets_bool)
    intersection = torch.logical_and(predictions_bool, targets_bool)
    return torch.sum(intersection)/torch.sum(union)

def mean_iou(predictions, targets, no_classes = 23):
    acc_iou = 0
    for class_id in range(no_classes):
        acc_iou += iou_single_class(predictions, targets, class_id)
    return acc_iou/no_classes

def iou_separate(predictions, targets, class_id):
    predictions_max = predictions.argmax(1)
    predictions_bool = predictions_max == class_id
    targets_bool = targets == class_id
    union = torch.logical_or(predictions_bool, targets_bool)
    intersection = torch.logical_and(predictions_bool, targets_bool)
    return torch.sum(intersection), torch.sum(union)
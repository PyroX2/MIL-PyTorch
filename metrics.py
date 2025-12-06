import torch
from torcheval.metrics import (
    BinaryAccuracy, BinaryF1Score, BinaryAUPRC, BinaryAUROC, BinaryConfusionMatrix, BinaryPrecision, BinaryRecall,
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUPRC, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall
)
from typing import List, Tuple


class BinaryMetricsCalculator:
    def __init__(self, threshold: float = 0.5) -> None:
        self.accuracy = BinaryAccuracy(threshold=threshold)
        self.f1_score = BinaryF1Score(threshold=threshold)
        self.auprc = BinaryAUPRC()
        self.auroc = BinaryAUROC()
        self.confusion_matrix = BinaryConfusionMatrix(threshold=threshold)
        self.precision = BinaryPrecision(threshold=threshold)
        self.recall = BinaryRecall(threshold=threshold)
    
    def calculate(self, outputs: List | torch.Tensor, targets: List | torch.Tensor) -> Tuple:
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(outputs, dtype=torch.float32)
        else:
            outputs = outputs.to(torch.float32)
            
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = targets.to(torch.float32)

        self.accuracy.update(outputs, targets)
        accuracy = self.accuracy.compute()

        self.f1_score.update(outputs, targets)
        f1_score = self.f1_score.compute()

        self.auprc.update(outputs, targets)
        auprc = self.auprc.compute()

        self.auroc.update(outputs, targets)
        auroc = self.auroc.compute()

        self.precision.update(outputs, targets)
        precision = self.precision.compute()

        self.recall.update(outputs, targets.to(torch.long))
        recall = self.recall.compute()

        self.confusion_matrix.update(outputs, targets.to(torch.long))
        confusion_matrix = self.confusion_matrix.compute()

        self.accuracy.reset()
        self.f1_score.reset()
        self.auprc.reset()
        self.auroc.reset()
        self.precision.reset()
        self.recall.reset()
        self.confusion_matrix.reset()

        return accuracy, f1_score, auprc, auroc, precision, recall, confusion_matrix


class MulticlassMetricsCalculator:
    def __init__(self, num_classes: int, avg_method: str = "micro") -> None:
        assert avg_method in ["micro", "macro", "None", None], f'avg method should be in ["micro", "macro", "None", None], found {avg_method}'

        if avg_method == "None":
            avg_method = None

        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average=avg_method)
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average=avg_method)
        self.auprc = MulticlassAUPRC(num_classes=num_classes, average=avg_method)
        self.auroc = MulticlassAUROC(num_classes=num_classes, average=avg_method)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.precision = MulticlassPrecision(num_classes=num_classes, average=avg_method)
        self.recall = MulticlassRecall(num_classes=num_classes, average=avg_method)
    
    def calculate(self, outputs: List | torch.Tensor, targets: List | torch.Tensor) -> Tuple:
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(outputs, dtype=torch.long)
        else:
            outputs = outputs.to(torch.long)
            
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.long)
        else:
            targets = targets.to(torch.long)

        self.accuracy.update(outputs, targets)
        accuracy = self.accuracy.compute()

        self.f1_score.update(outputs, targets)
        f1_score = self.f1_score.compute()

        self.auprc.update(outputs, targets)
        auprc = self.auprc.compute()

        self.auroc.update(outputs, targets)
        auroc = self.auroc.compute()

        self.precision.update(outputs, targets)
        precision = self.precision.compute()

        self.recall.update(outputs, targets)
        recall = self.recall.compute()

        self.confusion_matrix.update(outputs, targets.to(torch.long))
        confusion_matrix = self.confusion_matrix.compute()

        self.accuracy.reset()
        self.f1_score.reset()
        self.auprc.reset()
        self.auroc.reset()
        self.precision.reset()
        self.recall.reset()
        self.confusion_matrix.reset()

        return accuracy, f1_score, auprc, auroc, precision, recall, confusion_matrix
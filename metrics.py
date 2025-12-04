import torch
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryAUPRC, BinaryAUROC, BinaryConfusionMatrix, BinaryPrecision, BinaryRecall
from typing import List, Tuple


class BinaryMetricsCalculator:
    def __init__(self):
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.auprc = BinaryAUPRC()
        self.auroc = BinaryAUROC()
        self.confusion_matrix = BinaryConfusionMatrix()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
    
    def calculate(self, outputs: List, targets: List) -> Tuple:
        outputs = torch.tensor(outputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

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
    def __init__(self, num_classes):
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.auprc = BinaryAUPRC()
        self.auroc = BinaryAUROC()
        self.confusion_matrix = BinaryConfusionMatrix()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
    
    def calculate(self, outputs: List, targets: List) -> Tuple:
        outputs = torch.tensor(outputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

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
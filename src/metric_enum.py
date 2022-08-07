from enum import Enum


class MetricEnum(Enum):
    """Enum class with evaluation metrics."""
    PR_AUC = 'pr_auc'
    ROC_AUC = 'roc_auc'
    PRECISION = 'precision'
    ACCURACY = 'accuracy'

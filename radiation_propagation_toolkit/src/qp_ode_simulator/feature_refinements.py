"""Serializable observable-only mixture for feature-ablation experiments."""
import numpy as np


class WeakSignalGate:
    """Classifier estimates weak-event probability; query truth is never consumed."""
    def __init__(self, classifier, parent_svr, parent_et, center, expert=None, fraction=1., power=1):
        self.classifier = classifier
        self.parent_svr = parent_svr
        self.parent_et = parent_et
        self.center = np.asarray(center)
        self.expert = expert
        self.fraction = fraction
        self.power = power

    def predict(self, x):
        base = .5*(self.parent_svr.predict(x)+self.parent_et.predict(x))
        index = list(self.classifier.classes_).index(1)
        probability = self.classifier.predict_proba(x)[:, index]
        weight = self.fraction * probability**self.power
        target = self.center if self.expert is None else self.expert.predict(x)
        return base + weight[:, None]*(target-base)

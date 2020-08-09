import numpy as np


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def metrics_fn(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}


def prediction(logit):
    return np.argmax(logit, axis=1)
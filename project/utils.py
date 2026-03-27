import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

def align_predictions(predictions, labels, label_list):
    preds = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(preds, labels):
        curr_preds = []
        curr_labels = []

        for p, l in zip(pred, lab):
            if l != -100:
                curr_preds.append(label_list[p])
                curr_labels.append(label_list[l])

        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)

    return true_predictions, true_labels


def compute_metrics(predictions, labels, label_list):
    true_preds, true_labels = align_predictions(predictions, labels, label_list)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }
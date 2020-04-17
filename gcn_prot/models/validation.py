"""Calculate statistics for the validation dataset."""
import torch
from sklearn.metrics import confusion_matrix

from .train import forward_step


class Validation:
    """Run the validation and compute statistics."""

    def __init__(self, trained_model, valid):
        """Initialize."""
        self.model = trained_model
        self.prediction = []
        self.truth = []
        self.valid = valid
        self.stats = None

    def __str__(self):
        """Print in LaTeX tab format if statistics were computed."""
        if self.stats:
            return r"""\begin{table}[]
\centering
\begin{tabular}{llll}
\hline
\multicolumn{1}{|l|}{\textbf{Recall}} & \multicolumn{1}{l|}{\textbf{Precision}} & \multicolumn{1}{l|}{\textbf{Accuracy}} & \multicolumn{1}{l|}{\textbf{F-score}} \\ \hline
Strain 1                              & 140                                     & 1390648                                & 149577
\end{tabular}
\caption{}
\label{tab:my-table}
\end{table}"""
        else:
            return "Statistics not computed."

    def validate(self):
        """Run the pass forward trough the model."""
        self.model.eval()
        for batch in torch.utils.data.DataLoader(
            self.valid, shuffle=False, batch_size=1, drop_last=False
        ):
            pred, y = forward_step(batch, self.model, False)
            pred = torch.where(pred[0] == pred[0].max())[0]
            self.prediction.append(pred[0].cpu().tolist())
            self.truth.append(y[0].cpu().tolist())

    def compute_stats(self):
        """Confussion matrix and related statistics."""
        tn, fp, fn, tp = confusion_matrix(self.truth, self.prediction).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / len(self.valid)
        f_score = 2 * recall * precision / (recall + precision)
        self.stats = {
            "recall": recall,
            "precision": precision,
            "accuracy": accuracy,
            "f_score": f_score,
        }
        return self.stats

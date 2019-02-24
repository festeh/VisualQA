import torch

from ignite.metrics.accuracy import _BaseClassification


class VisualQAAccuracy(_BaseClassification):
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output

        assert y.size(1) == 10

        indices = torch.argmax(y_pred, dim=1).view(-1, 1)

        correct = torch.eq(indices, y).sum(1).type(torch.float32) / 3
        correct = torch.clamp(correct, 0, 1)

        self._num_correct += correct.sum().item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise ValueError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CrossEntropyLoss', 'SoftEntropyLoss']

class CrossEntropyLoss(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.
	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.
	Args:
	num_classes (int): number of classes.
	epsilon (float): weight.
	"""

	def __init__(
		self,
		num_classes,
		epsilon=0.1
	):
		super(CrossEntropyLoss, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon

		self.logsoftmax = nn.LogSoftmax(dim=1)
		assert(self.num_classes>0)

	def forward(self, results, targets):
		"""
		Args:
		inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
		targets: ground truth labels with shape (num_classes)
		"""

		inputs = results['prob']

		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

		loss = (- targets * log_probs).mean(0).sum()
		return loss

class SoftEntropyLoss(nn.Module):
	def __init__(
		self
	):
		super(SoftEntropyLoss, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, results, results_mean):
		assert (results_mean is not None)

		inputs = results['prob']
		targets = results_mean['prob']

		log_probs = self.logsoftmax(inputs)
		loss = (- self.softmax(targets).detach() * log_probs).mean(0).sum()
		return loss

import numpy as np


class SoftmaxRegression:
  def __init__(self, training_data, training_labels, learning_rate=None):
    self._training_data = np.array(training_data, dtype=float)
    self.training_labels = np.array(training_labels)
    self._classification_classes = None

    self.weight_vectors = np.ones((len(self.training_data[0]), len(self.classification_classes)), dtype=float)
    self.learning_rate = 1 if learning_rate is None else learning_rate

  @property
  def training_data(self):
    return np.array([np.hstack(([1], image), dtype=float) for image in self._training_data])

  @property
  def classification_classes(self):
    if self._classification_classes is None:
      self._classification_classes = list(set(tuple(x.astype(int)) for x in self.training_labels))
    return self._classification_classes

  @property
  def predicted_probabilities(self):
    Z = np.dot(self.training_data, self.weight_vectors)
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

  @property
  def gradient(self):
    n = len(self.training_data)
    return (1 / n) * np.dot(self.training_data.T, (self.predicted_probabilities - self.training_labels))

  @property
  def argmax(self):
    '''
    Returns a Numpy array of the argmax of each row of the predicted probabilities array.
    '''
    return np.argmax(self.predicted_probabilities, axis=1)

  def run(self, iterations):
    for k in range(iterations):
      self.weight_vectors -= self.learning_rate * self.gradient
    return None

  def classify(self, index_mapping):
    return np.array([index_mapping[index] for index in self.argmax])

  def apply_mapping(self, mapping):
      return np.array([mapping[tuple(x.astype(int).tolist())] for x in self.training_labels])

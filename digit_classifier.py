import numpy as np
from softmax_regression import SoftmaxRegression


class DigitClassifier(SoftmaxRegression):
  ONEHOT_TO_DIGIT_MAPPING = {tuple(np.eye(10, dtype=int)[i].tolist()): i for i in range(len(np.eye(10, dtype=int)))}
  DIGIT_TO_ONEHOT_MAPPING = {value: key for key, value in ONEHOT_TO_DIGIT_MAPPING.items()}
  INDEX_TO_DIGIT_MAPPING = {i:i for i in range(10)}

  def __init__(self, training_data, training_labels, learning_rate=None):
    super().__init__(training_data, training_labels, learning_rate)

  def classify(self):
    return super().classify(DigitClassifier.INDEX_TO_DIGIT_MAPPING)

  @property
  def confusion_matrix(self):
    # The Confusion Matrix
    mat = np.zeros((len(self.classification_classes), len(self.classification_classes)), dtype=int)
    predicted_classes = self.classify()
    actual_classes = np.array([DigitClassifier.ONEHOT_TO_DIGIT_MAPPING[tuple(self.training_labels[i].astype(int).tolist())] for i in range(len(self.training_labels))])
    mapped_classification_classes = [DigitClassifier.ONEHOT_TO_DIGIT_MAPPING[x] for x in self.classification_classes]
    # The indexes of the predicted/actual classes in the classification classes array.
    predicted_class_indexes = np.array([mapped_classification_classes.index(predicted_class) for predicted_class in predicted_classes])
    actual_class_indexes = np.array([mapped_classification_classes.index(actual_class) for actual_class in actual_classes])

    for i in range(len(actual_class_indexes)):
      actual_class_index = actual_class_indexes[i]
      predicted_class_index = predicted_class_indexes[i]
      mat[actual_class_index][predicted_class_index] += 1
    return mat

  @property
  def error_rate(self):
    return 1 - self.accuracy

  @property
  def accuracy(self):
    return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

import pickle

from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.metrics import f1_score, log_loss


class RandomForestClassifier:
    def __init__(self, max_depth=1000, verbose=2, model=None):
        if model:
            self.model = model
        else:
            self.model = RFClassifier(max_depth=max_depth, verbose=verbose)

    def fit(self, x, y, save=False, filename=None):
        """
        :param x: train input
        :param y: train labels
        :param save: OPTIONAL model saving
        :param filename: OPTIONAL saved model filename
        :return: None
        """
        self.model.fit(x, y)
        if save:
            if filename:
                pickle.dump(self.model, open(filename, "wb"))
            else:
                raise ValueError("A filename is required to save the model.")

    def accuracy(self, x, y):
        """
        :param x: test input
        :param y: test labels
        :return: model accuracy
        """
        return self.model.score(x, y)

    def predictions(self, x):
        """
        :param x: test input
        :return: vector of predicted labels
        """
        return self.model.predict(x)

    def log_loss(self, y_test, y_pred):
        """
        :param y_test: true labels
        :param y_pred: predicted labels
        :return: logarithmic loss
        """
        return log_loss(y_test, y_pred)

    def f1score(self, y_test, y_pred, average='macro'):
        """
        :param y_test: true labels
        :param y_pred: predicted labels
        :param average: OPTIONAL average strategy
        :return: f1 score
        """
        return f1_score(y_test, y_pred, average=average)

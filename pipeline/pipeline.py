#! /usr/bin/env python3

class ModelCase:
    y_true = None

    def __init__(self, model, X_tst, y_tst, y_true=None):
        """
        Initialize the model to be studied

        :param model: model to inspect
        :param X_tst: test data
        :param y_tst: test labels (possibly)
        """
        self.model = model
        self.X_tst = X_tst
        self.y_tst = y_tst

    


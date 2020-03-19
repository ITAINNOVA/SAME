import abc


class AbstractClassifier:
    """ Abstract class with specific methods for classifier models (training, validation and test) """

    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, config, train_data):

        """
        Classifier training.
        :param config: Model configuration.
        :param train_data: Train dataset with the textual information of each item and its label.
        :return: A model trained with train_data according to config.
        """
        pass

    @abc.abstractmethod
    def validation(self, config, val_data):
        """

        :param config: Model configuration.
        :param val_data: Validation dataset with the textual information of each item and its label.
        :return: Validation metrics
        """
        pass

    @abc.abstractmethod
    def test(self, config, test_data):

        """
        Classifier testing.
        :param config: Model configuration.
        :param test_data: Test dataset with the textual information of each item and its label.
        :return: Predictions of the model in the test_data, according to config.
        """

        pass

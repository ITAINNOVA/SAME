import abc


class AbstractRecommender:

    """ Abstract class with the specific methods of a recommender """

    def __init__(self):
        pass

    @abc.abstractmethod
    def recommend(self, user_id, how_many):

        """
        Recommends the best items for a specific user.

        :param user_id: Id of the user to recommend.
        :param how_many: Number of items that we recommend to the specific user.
        :return: Id of the items that the recommender returns.
        """

        pass

    @abc.abstractmethod
    def estimate_preference(self, user_id, item_id):

        """
        Estimate the preference value by a specific user.

        :param user_id: Id of the user to recommend.
        :param item_id: Id of the item to recommend.
        :return: The estimate preference by the sepecific recommender.
        """
        pass

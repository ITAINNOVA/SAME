from semantic_aware_models.models.recommendation.abstract_recommender import AbstractRecommender
from semantic_aware_models.dataset.movielens.movielens_data_model import *
from surprise import NormalPredictor
from surprise.reader import Reader
from surprise.dataset import Dataset
import time



class RandomRecommender(AbstractRecommender):

    """ Algorithm predicting a random rating based on the distribution of the training set, which is assumed to be normal. """

    def __init__(self, ratings_file_path=None, separator=None):
        super(AbstractRecommender, self).__init__()
        # Create the recommendation input_model and configure its input parameters:
        self.model = NormalPredictor()
        self.rating_data_model = RatingDataModel(ratings_file_path=ratings_file_path, separator=separator)
        self.separator = separator

    def recommend(self, user_id, how_many):

        """
        Recommends the best items for a specific user.
        :param user_id: Id of the user to recommend.
        :param how_many: Number of items that we recommend to the specific user.
        :return: Id of the items that the recommender returns.
        """

        # Items not seen by a specific user.
        item_ids_not_seen_from_user = self.rating_data_model.get_item_ids_not_seen_from_user(user_id)

        list_recommend = []
        for item_id in item_ids_not_seen_from_user:
            preference = self.estimate_preference(user_id, item_id)
            list_recommend.append([item_id, preference])
            print(item_id, ', ', preference)

        list_recommend.sort(key=lambda x: x[1], reverse=True)

        return list_recommend[:how_many]

    def estimate_preference(self, user_id, item_id):

        """
        Estimate the preference value by a specific user.
        :param user_id: Id of the user to recommend.
        :param item_id: Id of the item to recommend.
        :return: The estimate preference by the sepecific recommender.
        """

        # train file:
        df_ratings = self.rating_data_model.df_ratings
        # A reader is still needed but only the rating_scale param is requiered.
        reader = Reader(rating_scale=(self.rating_data_model.get_min_preference(), self.rating_data_model.get_max_preference()))
        train_data = Dataset(reader=reader)
        # The columns must correspond to user id, item id and ratings (in that order).
        raw_trainset = train_data.load_from_df(df_ratings[['user_id', 'item_id', 'rating']], reader)
        trainset = train_data.construct_trainset(raw_trainset.raw_ratings)

        # Train recommendation input_model:
        self.model.fit(trainset)

        return float(self.model.estimate(u=user_id, i=item_id)[0])

    def recommend_rival(self, n_folds, train_test_file_path, reader, recommendation_file_path):

        """
        Prepare the predictions to take them to RiVaL Toolkit.
        :param n_folds: Number of folds.
        :param train_test_file_path: Path with train and input_test files.
        :param recommendation_file_path: Path where the suitable files to run RiVaL Toolkit are saved.
        :return: The suitable files to run RiVaL Toolkit are saved.
        """

        for i in range(n_folds):
            print('Fold: ', i)

            timestart = time.time()
            # train file:
            train_file_name = train_test_file_path + 'train_bin_verified_sep_' + str(i) + '.csv'
            train_data = Dataset(reader=reader)
            raw_trainset = train_data.read_ratings(file_name=train_file_name)
            trainset = train_data.construct_trainset(raw_trainset)
            timeend = time.time()
            print('Train file loading time: ', (timeend - timestart), 'seconds')

            timestart = time.time()
            # Train recommendation input_model:
            self.model.fit(trainset)
            timeend = time.time()
            print('Training time: ', (timeend - timestart), 'seconds')

            # input_test file:
            timestart = time.time()
            test_file_name = train_test_file_path + 'test_bin_verified_sep_' + str(i) + '.csv'
            test_data = Dataset(reader=reader)
            raw_testset = test_data.read_ratings(file_name=test_file_name)
            testset = test_data.construct_testset(raw_testset)
            timeend = time.time()
            print('Load time of the input_test file: ', (timeend - timestart), 'seconds')

            # Predictions:
            timestart = time.time()
            predictions = self.model.test(testset)
            file_name = open(recommendation_file_path + 'recs_' + str(i) + '.csv', 'w')
            for pred in predictions:
                user_id = pred[0]
                item_id = pred[1]
                rating_real = pred[2]
                rating_estimated = pred[3]
                file_name.write(user_id + "\t" + item_id + "\t" + str(rating_estimated) + '\n')
            timeend = time.time()
            print('Prediction time: ', (timeend - timestart), 'seconds')

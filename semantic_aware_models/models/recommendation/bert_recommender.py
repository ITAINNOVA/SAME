from semantic_aware_models.models.recommendation.abstract_recommender import AbstractRecommender
from semantic_aware_models.models.classification.bert_classifier import BertClassifier
from semantic_aware_models.dataset.movielens.movielens_data_model import *
from semantic_aware_models.utils.gpu import GPU
from semantic_aware_models.models.language.bert.inputs import InputExample
import os
import logging
import time

from semantic_aware_models.models.language.bert.data_processors import processors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename='bert_classifier.log', filemode='w')
gpu = GPU()


class BertRecommender(AbstractRecommender):

    """ Bert Recommender Class """

    def __init__(self, ratings_file_path, items_file_path, config_model):
        super(BertRecommender, self).__init__()
        self.results = {}

        self.bert_cbrs = BertClassifier()

        # Load data
        self.descriptions = ItemUnstructuredDataModel(items_file_path, separator='::')
        self.ratings_file_path = ratings_file_path
        self.ratings_data_model = RatingDataModel(ratings_file_path=ratings_file_path, separator='::')
        self.items_unstructured_columns = ['item_id', 'title', 'description']
        self.items_all = self.descriptions.get_item_ids()

        # Load Configuration

        self.config_model = config_model
        self.processor = processors[self.config_model['task_name']]()



        self.device = gpu.get_default_device()

    def recommend(self, user_id, how_many):

        """
        Recommends the best items for a specific user.
        :param user_id: Id of the user to recommend.
        :param how_many: Number of items that we recommend to the specific user.
        :return: Id of the items that the recommender returns.
        """

        item_ids_not_seen_from_user = self.ratings_data_model.get_item_ids_not_seen_from_user(user_id, self.items_all)
        # print('item_ids_not_seen_from_user:', item_ids_not_seen_from_user)

        list_recommend = []
        for item_id in item_ids_not_seen_from_user:
            preference = self.estimate_preference(user_id, item_id)
            list_recommend.append([item_id, preference])

        list_recommend.sort(key=lambda x: x[1], reverse=True)

        return list_recommend[:how_many]

    def estimate_preference(self, user_id, item_id):

        """
        Estimate the preference value by a specific user.
        :param user_id: Id of the user to recommend.
        :param item_id: Id of the item to recommend.
        :return: The estimate preference by the sepecific recommender.
        """

        if not os.path.isfile(self.config_model['output_dir']):

            data = self.processor.get_train_examples(descriptions=self.descriptions, ratings=self.ratings_data_model,
                                                user_id=user_id)
            self.bert_cbrs.train_model(config=self.config_model, train_data=data)

        # TEST
        examples = []
        description = self.descriptions.get_description_from_id(item_id=item_id)
        rating = self.ratings_data_model.get_preference_value(user_id=user_id, item_id=item_id)
        examples.append(InputExample(guid=item_id, text_a=description, text_b=None, label=rating))

        result = self.bert_cbrs.test_model(config=self.config_model, test_data=examples)


        return (float(result[0]) + 1.0)

    def __estimate_preference_rival(self, train_data, test_data):

        # TRAIN

        timestart = time.time()

        self.bert_cbrs.train_model(config=self.config_model, train_data=train_data)

        timeend = time.time()

        train_time = timeend - timestart

        # TEST

        timestart = time.time()

        preds = self.bert_cbrs.test_model(config=self.config_model, test_data=test_data)

        timeend = time.time()

        test_time = timeend - timestart

        return preds, train_time, test_time



    def recommend_rival(self, n_folds, train_test_file_path, recommendation_file_path):

        """
        Prepare the predictions to take them to RiVaL Toolkit.
        :param n_folds: Number of folds.
        :param train_test_file_path: Path with train and input_test files.
        :param recommendation_file_path: Path where the suitable files to run RiVaL Toolkit are saved.
        :return: The suitable files to run RiVaL Toolkit are saved.
        """

        for i in range(n_folds):

            test_file_name = train_test_file_path + 'test_bin_verified_sep_' + str(i) + '.csv'
            train_file_name = train_test_file_path + 'train_bin_verified_sep_' + str(i) + '.csv'

            ratings_train_data_model = RatingDataModel(ratings_file_path=train_file_name, separator="	")
            ratings_test_data_model = RatingDataModel(ratings_file_path=test_file_name, separator="	")


            file_name = open(recommendation_file_path + 'recs_' + str(i) + '.csv', 'w')
            user_ids = ratings_test_data_model.get_user_ids()
            for user_id in user_ids:

                train_data = self.processor.get_train_examples(descriptions=self.descriptions,
                                                          ratings=ratings_train_data_model,
                                                          user_id=user_id)

                test_data = self.processor.get_train_examples(descriptions=self.descriptions,
                                                              ratings=ratings_test_data_model,
                                                                user_id=user_id)

                rating_estimated_list, time_train, time_test = self.__estimate_preference_rival(train_data=train_data,
                                                                                              test_data=test_data)


                print(i,';', user_id, ';', time_train, ";", time_test)

                items_ids = ratings_test_data_model.get_item_ids_from_user(user_id)
                j=0
                for i in items_ids:
                    file_name.write(str(user_id) + "\t" + str(i) + "\t" + str(float(rating_estimated_list[j]) + 1.0) + '\n')
                    j+=1

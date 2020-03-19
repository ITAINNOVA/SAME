from semantic_aware_models.models.recommendation.abstract_recommender import AbstractRecommender
from semantic_aware_models.dataset.movielens.movielens_data_model import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

import nltk
nltk.download('punkt')
import string
import time



class ContentBasedRecommender(AbstractRecommender):

    """ A basic content-based recommendation algorithm, by using Vector Space Model. """

    def __init__(self, separator, items_file_path, ratings_file_path=None):
        super(AbstractRecommender, self).__init__()
        self.separator = separator
        self.results = {}
        self.items_file_path=items_file_path

        if ratings_file_path is None:
            self.rating_data_model = None
            self.test_data_model = None
        else:
            self.rating_data_model = RatingDataModel(ratings_file_path=ratings_file_path, separator=separator)

        self.the_tings = [('items-translated', 'item_name_translated', 50),('item_categories-translated','item_category_name_translated', 10),
        ('shops-translated','shop_name_translated', 10)]

        self.trans_table = {ord(c): None for c in string.punctuation + string.digits}
        self.stemmer = PorterStemmer()
        timestart = time.time()
        self.__get_similarity_map(items_file_path)
        timeend = time.time()
        # print('Time to calculate the matrix of similarities between all items: ', (timeend - timestart), 'seconds')

    def recommend(self, user_id, how_many):

        """
        Recommends the best items for a specific user.
        :param user_id: Id of the user to recommend.
        :param how_many: Number of items that we recommend to the specific user.
        :return: Id of the items that the recommender returns.
        """

        # Items not seen by a specific user.
        self.separator = '::'
        item_data_model = ItemUnstructuredDataModel(self.items_file_path, self.separator)
        items_all = item_data_model.get_item_ids()
        item_ids_not_seen_from_user = self.rating_data_model.get_item_ids_not_seen_from_user(user_id, items_all)

        list_recommend = []
        for item_id in item_ids_not_seen_from_user:
            preference = self.estimate_preference(user_id, item_id)
            list_recommend.append([item_id, preference])
        # print('unsort_list_recommend: ', list_recommend)

        list_recommend.sort(key=lambda x: x[1], reverse=True)
        # print('list_recommend: ', list_recommend)

        return list_recommend[:how_many]

    def estimate_preference(self, user_id, item_id):

        """
        Estimate the preference value by a specific user.
        :param user_id: Id of the user to recommend.
        :param item_id: Id of the item to recommend.
        :return: The estimate preference by the sepecific recommender.
        """

        # Items seen by a specific user: Train
        items_ids_from_user = self.rating_data_model.get_item_ids_from_user(user_id)

        result = float(0.0)
        if item_id in self.results:
            # Items most similar to the user:
            items_most_similar = self.results[item_id]

            preference = 0.0
            total_similarity = 0.0
            for it_sim in items_most_similar:
                score_similarity = it_sim[0]
                id = it_sim[1]
                if id in items_ids_from_user:
                    rating = self.rating_data_model.get_preference_value(user_id, id)
                    preference += score_similarity*rating
                    total_similarity += abs(score_similarity)
            if total_similarity != 0.0:
                result = float(preference / total_similarity)
        return result

    def recommend_rival(self, n_folds, train_test_file_path, recommendation_file_path):

        """
        Prepare the predictions to take them to RiVaL Toolkit.
        :param n_folds: Number of folds.
        :param train_test_file_path: Path with train and input_test files.
        :param recommendation_file_path: Path where the suitable files to run RiVaL Toolkit are saved.
        :return: The suitable files to run RiVaL Toolkit are saved.
        """

        for i in range(n_folds):
            print('Fold: ', i)
            # input_test and train files:
            timestart = time.time()
            test_file_name = train_test_file_path + 'test_bin_verified_sep_' + str(i) + '.csv'
            self.test_data_model = RatingDataModel(ratings_file_path=test_file_name, separator='\t')
            train_file_name = train_test_file_path + 'train_bin_verified_sep_' + str(i) + '.csv'
            self.rating_data_model = RatingDataModel(ratings_file_path=train_file_name, separator='\t')
            timeend = time.time()
            # print('Time to load the DF of ratings (from input_test): ', (timeend - timestart), 'seconds')

            timestart = time.time()
            file_name = open(recommendation_file_path + 'recs_' + str(i) + '.csv', 'w')
            user_ids = self.test_data_model.get_user_ids()
            for user_id in user_ids:
                timestart_user = time.time()
                # Items not seen by user: Test
                item_ids = self.test_data_model.get_item_ids_from_user(user_id)
                # print('item_ids: ', item_ids)

                for item_id in item_ids:
                    rating_estimated = self.estimate_preference(user_id, item_id)
                    file_name.write(str(user_id) + "\t" + str(item_id) + "\t" + str(rating_estimated) + '\n')
                    print(str(user_id) + "\t" + str(item_id) + "\t" + str(rating_estimated) + '\n')
                timeend_user = time.time()
                print(i, ';', user_id, ';', timeend_user - timestart_user)

            timeend = time.time()
            # print('Total prediction time for all users: ', (timeend - timestart), 'seconds')




    # Gets a map with all the similarities between the users.

    def __tokenize(self,text):

        tokens = [word for word in nltk.word_tokenize(text.translate(self.trans_table)) if len(word) > 1]
        stems = [self.stemmer.stem(item) for item in tokens]
        return stems



    def __get_similarity_map(self, items_file_path):
        # Read item file: movies_unstructured_test.dat
        self.separator = '::'
        item_data_model = ItemUnstructuredDataModel(items_file_path, self.separator)
        # Gets all descriptions in the input_model, in order.
        description_items = item_data_model.get_description()

        # Get the matrix TF-IDF.
        # tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tf = TfidfVectorizer(analyzer='word', tokenizer= self.__tokenize, min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(description_items)

        # Obtain the similarities between the textual descriptions.
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Sort the most similar items by item.
        df_unstructured_items = item_data_model.df_items
        for idx, row in df_unstructured_items.iterrows():

            max_value = (len(description_items)+1)
            similar_indices = cosine_similarities[idx].argsort()[:-max_value:-1]
            similar_items = [(cosine_similarities[idx][i], df_unstructured_items['item_id'][i]) for i in similar_indices]

            # First item is the item itself, so remove it.
            self.results[row['item_id']] = similar_items[1:]
            # print(self.results)
        # print('result_total: ', self.results)






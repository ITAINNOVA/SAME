from semantic_aware_models.models.recommendation.abstract_recommender import AbstractRecommender
from semantic_aware_models.models.classification.brnn_classifier import DeepCBRS
from semantic_aware_models.dataset.movielens.movielens_data_model import *
from semantic_aware_models.dataset.movielens.deep_cbrs_data_model import DeepCBRSDataModel
from semantic_aware_models.utils.gpu import GPU
from semantic_aware_models.models.language.bert.early_stopping import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

gpu = GPU()


class DeepCBRSRecommender(AbstractRecommender):
    """
        This implementation is loosely based on their LUA's homonym published at https://github.com/nlp-deepcbrs/amar
    """
    def __init__(self, separator, ratings_file_path, structured_file_path, unstructured_file_path, models_path):
        super(AbstractRecommender, self).__init__()
        self.separator = separator
        self.results = {}
        self.path_models = models_path

        # Load data
        self.data_model = DeepCBRSDataModel()
        self.items_data = self.data_model.read_items_data(train_path=unstructured_file_path)
        self.genres_data = self.data_model.load_items_genres(genres_filename=structured_file_path,
                                                             item2pos=self.items_data['item2pos'])
        self.authors_data = self.data_model.load_items_authors(authors_filename=structured_file_path,
                                                             item2pos=self.items_data['item2pos'])
        self.directors_data = self.data_model.load_items_directors(directors_filename=structured_file_path,
                                                         item2pos=self.items_data['item2pos'])

        self.wiki_categories_data = self.data_model.load_items_wiki_categories(wiki_categories_filename=structured_file_path,
                                                                 item2pos=self.items_data['item2pos'])

        self.ratings_file_path = ratings_file_path
        self.rating_data_model = RatingDataModel(ratings_file_path=ratings_file_path, separator='::')
        self.items_unstructured_columns = ['item_id', 'title', 'description']
        self.item_ids = ItemDataModel(items_file_path=unstructured_file_path, separator='::', items_columns=self.items_unstructured_columns)
        self.items_all = self.item_ids.get_item_ids()

        #Build Padds
        self.items_data['items'] = self.data_model.pad_items_data(self.items_data)
        self.genres_data['genres'] = self.data_model.pad_genres_data(self.genres_data)
        self.authors_data['authors'] = self.data_model.pad_authors_data(self.authors_data)
        self.directors_data['directors'] = self.data_model.pad_directors_data(self.directors_data)
        self.wiki_categories_data['wiki_categories'] = self.data_model.pad_wiki_categories_data(self.wiki_categories_data)

        self.batch_size = 5

        self.device = gpu.get_default_device()

    def recommend(self, user_id, how_many):
        """
        :param user_id: Id of the user to recommend.
        :param how_many: Number of items that we recommend to the specific user.
        :return: Id of the items that the recommender returns.
        """

        # Items not seen by a specific user.
        item_ids_not_seen_from_user = self.rating_data_model.get_item_ids_not_seen_from_user(user_id, self.items_all)
        print('item_ids_not_seen_from_user:', item_ids_not_seen_from_user)

        list_recommend = []
        for item_id in item_ids_not_seen_from_user:
            preference = self.estimate_preference(user_id, item_id)
            list_recommend.append([item_id, preference])

        list_recommend.sort(key=lambda x: x[1], reverse=True)

        return list_recommend[:how_many]

    def estimate_preference(self, user_id, item_id):

        """
        :param user_id: Id of the user to recommend.
        :param item_id: Id of the item to recommend.
        :return: The estimate preference by the sepecific recommender.
        """

        path = self.path_models + '\model_user_{}.pt'.format(user_id)
        if not os.path.isfile(path):
            self.__train_network(user_id=user_id, train_path=self.ratings_file_path, save_model_path=path)

        model = DeepCBRS(items_data=self.items_data, genres_data=self.genres_data, authors_data=self.authors_data,
                         directors_data=self.directors_data, wiki_categories_data=self.wiki_categories_data, batch_size=1)
        model.load_state_dict(torch.load(path))
        model.eval()

        ids_samples = []
        ids_samples.append(item_id)
        item_tensors = tuple([self.items_data['items'][x] for x in ids_samples])
        items_input = item_tensors[0]

        genres_tensors = tuple([self.genres_data['genres'][x] for x in ids_samples])
        genres_input = torch.unsqueeze(genres_tensors[0],0)

        authors_tensors = tuple([self.authors_data['authors'][x] for x in ids_samples])
        authors_input = torch.unsqueeze(authors_tensors[0],0)

        directors_tensors = tuple([self.directors_data['directors'][x] for x in ids_samples])
        directors_input = torch.unsqueeze(directors_tensors[0],0)

        wiki_categories_tensors = tuple([self.wiki_categories_data['wiki_categories'][x] for x in ids_samples])
        wiki_categories_input = torch.unsqueeze(wiki_categories_tensors[0],0)

        result = model.forward(items_input, genres_input, authors_input, directors_input, wiki_categories_input).item()

        return result

    def __estimate_preference_rival(self, user_id, item_id, train_path, save_path):
        if not os.path.isfile(save_path):
            self.__train_network(user_id=user_id, train_path=train_path, save_model_path=save_path)

        model = DeepCBRS(items_data=self.items_data, genres_data=self.genres_data, authors_data=self.authors_data,
                         directors_data=self.directors_data, wiki_categories_data=self.wiki_categories_data,
                         batch_size=1)

        model.to(self.device)
        model.load_state_dict(torch.load(save_path))
        model.eval()

        timestart = time.time()

        ids_samples = []
        ids_samples.append(item_id)
        item_tensors = tuple([self.items_data['items'][x] for x in ids_samples])
        items_input = item_tensors[0]
        items_input = gpu.to_device(items_input, self.device)

        genres_tensors = tuple([self.genres_data['genres'][x] for x in ids_samples])
        genres_input = torch.unsqueeze(genres_tensors[0], 0)
        genres_input = gpu.to_device(genres_input, self.device)

        authors_tensors = tuple([self.authors_data['authors'][x] for x in ids_samples])
        authors_input = torch.unsqueeze(authors_tensors[0], 0)
        authors_input = gpu.to_device(authors_input, self.device)

        directors_tensors = tuple([self.directors_data['directors'][x] for x in ids_samples])
        directors_input = torch.unsqueeze(directors_tensors[0], 0)
        directors_input = gpu.to_device(directors_input, self.device)

        wiki_categories_tensors = tuple([self.wiki_categories_data['wiki_categories'][x] for x in ids_samples])
        wiki_categories_input = torch.unsqueeze(wiki_categories_tensors[0], 0)
        wiki_categories_input = gpu.to_device(wiki_categories_input, self.device)

        predictions = model.forward(items_input, genres_input, authors_input, directors_input, wiki_categories_input)
        result = torch.max(predictions, 1)[1].item()+1
        print(result)
        timeend = time.time()
        return result, (timeend - timestart)

    def recommend_rival(self, n_folds, train_test_file_path, recommendation_file_path):
        """
        :param n_folds: Number of folds.
        :param train_test_file_path: Path with train and input_test files.
        :param recommendation_file_path: Path where the suitable files to run RiVaL Toolkit are saved.
        :return: The suitable files to run RiVaL Toolkit are saved.
        """

        for i in range(n_folds):
            # input_test file:
            print('----------------- Fold {} -----------------'.format(i))
            test_file_name = train_test_file_path + 'test_bin_verified_sep_' + str(i) + '.csv'
            train_file_name = train_test_file_path + 'train_bin_verified_sep_' + str(i) + '.csv'
            rating_data_model = RatingDataModel(ratings_file_path=test_file_name, separator='	')

            file_name = open(recommendation_file_path + 'recs_' + str(i) + '.csv', 'w')
            user_ids = rating_data_model.get_user_ids()
            for user_id in user_ids:
                print('----------- User {} -----------'.format(user_id))
                # Items not seen by user:
                save_path = self.path_models + '\model_fold_{}'.format(i) + '_user_{}.pt'.format(user_id)
                item_ids = rating_data_model.get_item_ids_from_user(user_id)

                time = 0.0
                for item_id in item_ids:
                    rating_estimated, time = self.__estimate_preference_rival(user_id, item_id, train_file_name, save_path=save_path)
                    file_name.write(str(user_id) + "\t" + str(item_id) + "\t" + str(rating_estimated) + '\n')
                    time += time

                print('Tiempo de Predicción: ', (time), 'segundos')

    def __train_network(self, user_id, train_path, save_model_path):

        timestart = time.time()
        model = DeepCBRS(items_data=self.items_data, genres_data=self.genres_data, authors_data=self.authors_data,
                         directors_data=self.directors_data, batch_size=self.batch_size,
                         wiki_categories_data=self.wiki_categories_data)

        model.to(self.device)

        num_epochs = 10
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.000001)
        early_stopping = EarlyStopping(patience=5, verbose=False)

        # Training
        # Get the ids of the movies seen by the user fixed:
        rating_data_test = RatingDataModel(ratings_file_path=train_path, separator='	')
        list_ids = rating_data_test.get_item_ids_from_user(user_id=user_id)
        num_items = len(list_ids)
        indices = torch.tensor(list_ids)
        samples = list(indices.split(self.batch_size))
        samples.pop()

        for epoch in range(num_epochs):
            train_loss = 0
            for i in range(len(samples)):
                ids_samples = list()
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                for j in range(len(samples[i])):
                    ids_samples.append(samples[i][j].item())
                item_tensors = tuple([self.items_data['items'][x] for x in ids_samples])
                items_input = (torch.cat(item_tensors, 0)).view(self.batch_size, -1)
                items_input = gpu.to_device(items_input, self.device)

                genres_tensors = tuple([self.genres_data['genres'][x] for x in ids_samples])
                genres_input = (torch.cat(genres_tensors, 0)).view(self.batch_size, -1)
                genres_input = gpu.to_device(genres_input, self.device)

                authors_tensors = tuple([self.authors_data['authors'][x] for x in ids_samples])
                authors_input = (torch.cat(authors_tensors, 0)).view(self.batch_size, -1)
                authors_input = gpu.to_device(authors_input, self.device)

                directors_tensors = tuple([self.directors_data['directors'][x] for x in ids_samples])
                directors_input = (torch.cat(directors_tensors, 0)).view(self.batch_size, -1)
                directors_input = gpu.to_device(directors_input, self.device)

                wiki_categories_tensors = tuple([self.wiki_categories_data['wiki_categories'][x] for x in ids_samples])
                wiki_categories_input = (torch.cat(wiki_categories_tensors, 0)).view(self.batch_size, -1)
                wiki_categories_input = gpu.to_device(wiki_categories_input, self.device)

                list_ratings = [rating_data_test.get_preference_value(user_id=user_id, item_id=x)-1 for x in ids_samples]
                ratings = torch.tensor(list_ratings, dtype=torch.long)
                ratings = gpu.to_device(ratings, self.device)

                # Step 3. Run our forward pass.
                predictions = model.forward(items_input, genres_input, authors_input, directors_input,
                                            wiki_categories_input)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(predictions, ratings)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                print.info("Hit early stopping at epoch {}".format(epoch+1))
                break

        torch.save(model.state_dict(), save_model_path)

        timeend = time.time()
        print('Tiempo de Entrenamiento: ', (timeend - timestart), 'segundos')

        return model
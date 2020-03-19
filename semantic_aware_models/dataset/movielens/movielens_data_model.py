import pandas as pd

class UserDataModel:

    """ Class to get user-related information. """

    users_columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']

    def __init__(self, users_file_path, separator):
        """
        Create an object of the class.
        :param users_file_path: path of the users file.
        :param separator: file separator.
        """
        self.df_users = pd.read_table(users_file_path, sep=separator, names=self.users_columns, engine='python')
        self.df_users = self.df_users.sort_values(by=['user_id'])

    def get_num_users(self):

        """
        Gets total number of users known to the input_model.
        :return: number of users.
        """
        return len(self.df_users.index)

    def get_user_ids(self):

        """
        Gets all user IDs in the input_model, in order.
        :return: a list with user ids.
        """
        return self.df_users['user_id'].tolist()


    def get_gender(self):

        """
        Gets all genders in the input_model, in order.
        :return: a list with user genders.
        """
        return self.df_users['gender'].tolist()

    def get_age(self):

        """
        Gets all ages in the input_model, in order.
        :return: a list with user ages.
        """
        return self.df_users['age'].tolist()

    def get_occupation(self):

        """
        Gets all occupations in the input_model, in order.
        :return: a list with the users occupations.
        """
        return self.df_users['occupation'].tolist()

    def get_zip_code(self):

        """
        Gets all zip_codes in the input_model, in order.
        :return: a list with the users zip code.
        """
        return self.df_users['zip_code'].tolist()


class ItemDataModel:

    """ Class to get item-related information. """

    def __init__(self, items_file_path, separator, items_columns):

        """
        Create an object of the class.
        :param items_file_path: path with the items file.
        :param separator: file separator.
        :param items_columns: list with the name of the columns.
        """
        self.df_items = pd.read_csv(items_file_path, sep=separator, names=items_columns, engine='python')

        self.df_items = self.df_items.sort_values(by=['item_id'])

    def get_num_items(self):

        """
        Gets total number of items known to the input_model. This is generally the union
        of all items preferred by at least one user but could include more.
        """
        return len(self.df_items.index)

    def get_item_ids(self):

        """
        Gets all item IDs in the input_model, in order.
        """
        return self.df_items['item_id'].tolist()

    def get_title(self):

        """
        Gets all titles in the input_model, in order.
        """
        return self.df_items['title'].tolist()


class ItemStructuredDataModel(ItemDataModel):

    """ Class to get item-related structured information. """

    items_structured_columns = ['item_id', 'title', 'genres', 'budget', 'cinematography', 'director', 'distributor', 'editing', 'gross', 'music_composer', 'producer', 'runtime', 'starring', 'wiki_page_id', 'wiki_page_revision_id', 'writer', 'caption', 'country', 'language', 'studio', 'type', 'subject']

    def __init__(self, items_structured_file_path, separator):
        """
        Create an object of the class.
        :param items_file_path: path with the structured information file.
        :param separator: file separator.
        """
        super(ItemStructuredDataModel, self).__init__(items_file_path=items_structured_file_path, separator=separator, items_columns=self.items_structured_columns)

    def get_genres(self):

        """
        Gets all genres in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['genres'].tolist()

    def get_budget(self):

        """
        Gets all budgets in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['budget'].tolist()

    def get_cinematography(self):

        """
        Gets all cinematography in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['cinematography'].tolist()

    def get_director(self):

        """
        Gets all director in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['director'].tolist()


    def get_distributor(self):

        """
        Gets all distributor in the input_model, in order
        :return: A list with the requested information.
        """
        return self.df_items['distributor'].tolist()

    def get_editing(self):

        """
        Gets all editing in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['editing'].tolist()

    def get_gross(self):

        """
        Gets all gross in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['gross'].tolist()

    def get_music_composer(self):

        """
        Gets all music_composer in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['music_composer'].tolist()

    def get_producer(self):

        """
        Gets all producer in the input_model, in order
        :return: A list with the requested information.
        """
        return self.df_items['producer'].tolist()

    def get_runtime(self):

        """
        Gets all runtime in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['runtime'].tolist()

    def get_starring(self):

        """
        Gets all starring in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['starring'].tolist()

    def get_wiki_page_id(self):

        """
        Gets all wiki_page_id in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['wiki_page_id'].tolist()

    def get_wiki_page_revision_id(self):

        """
        Gets all wiki_page_revision_id in the input_model, in order.
        :return: A list with the requested information.
        """

        return self.df_items['wiki_page_revision_id'].tolist()

    def get_writer(self):

        """
        Gets all writer in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['writer'].tolist()

    def get_caption(self):

        """
        Gets all caption in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['caption'].tolist()

    def get_country(self):

        """
        Gets all country in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['country'].tolist()

    def get_language(self):

        """
        Gets all language in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['language'].tolist()

    def get_studio(self):

        """
        Gets all studio in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['studio'].tolist()

    def get_type(self):

        """
        Gets all type in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['type'].tolist()

    def get_comment(self):

        """
        Gets all comment in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['comment'].tolist()

    def get_subject(self):

        """
        Gets all subjects in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['subject'].tolist()


class ItemUnstructuredDataModel(ItemDataModel):

    """ Class to get item-related unstructured information. """

    items_unstructured_columns = ['item_id', 'title', 'description']

    def __init__(self, items_unstructured_file_path, separator):

        """
        Create an object of the class.
        :param items_unstructured_file_path: path with the unstructured information file.
        :param separator: file separator.
        """
        super(ItemUnstructuredDataModel, self).__init__(items_file_path=items_unstructured_file_path, separator=separator, items_columns=self.items_unstructured_columns)

    def get_description(self):

        """
        Gets all descriptions in the input_model, in order.
        :return: A list with the requested information.
        """
        return self.df_items['description'].tolist()

    def get_description_from_id(self, item_id):
        """
        Gets the description from a film.
        :param item_id: film id.
        :return: the film description.
        """
        return self.df_items[self.df_items['item_id'] == item_id]['description'].values[0]




class RatingDataModel:

    """ Class to get rating-related information. """

    ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']

    def __init__(self, ratings_file_path, separator):

        """
        Create an object of the class.
        :param ratings_file_path: path with the unstructured information file.
        :param separator: file separator.
        """
        self.df_ratings = pd.read_csv(ratings_file_path, sep=separator, names=self.ratings_columns, engine='python')

    def get_df_user_ratings_from(self, user_id):

        """
        Gets df_ratings of a specific user, ordered by item ID.
        :param user_id: id of the user.
        :return: A data frame with the requested information.
        """
        df_user_ratings = self.df_ratings[self.df_ratings['user_id'] == user_id]
        df_user_ratings = df_user_ratings.sort_values(by=['item_id'])
        return df_user_ratings

    def get_preferences_from_user(self, user_id):

        """
        Gets user's preferences, ordered by item ID.
        :param user_id: id of the user.
        :return: A list with the requested information.
        """
        df_user_ratings = self.get_df_user_ratings_from(user_id)
        return df_user_ratings['rating'].tolist()

    def get_item_ids_from_user(self, user_id):

        """
        Gets IDs of items user expresses a preference for.
        :param user_id: id of the user.
        :return: A list with the requested information.
        """
        df_user_ratings = self.get_df_user_ratings_from(user_id)
        return df_user_ratings['item_id'].tolist()

    def get_item_ids_not_seen_from_user(self, user_id, items_all):

        """
        Gets IDs of items not seen by a user.
        :param user_id: id of the user.
        :param items_all: all items of the file.
        :return: A list with the requested information.
        """

        items_seen = self.get_item_ids_from_user(user_id)
        items_seen = list(map(int, items_seen))  # Asegurar que sean de tipo int
        items_all = list(map(int, items_all))  # Asegurar que sean de tipo int
        return list(set(items_all) - set(items_seen))

    def get_df_item_ratings_from(self, item_id):

        """
        Gets df_ratings of a specific item, ordered by user ID.
        :param item_id: id of the item.
        :return: A data frame with the requested information.
        """
        df_item_ratings = self.df_ratings[self.df_ratings['item_id'] == item_id]
        df_item_ratings = df_item_ratings.sort_values(by=['user_id'])
        return df_item_ratings

    def get_preferences_for_item(self, item_id):

        """
        Gets all existing preferences expressed for that item, ordered by user ID, as an array.
        :param item_id: id of the item.
        :return: A list with the requested information.
        """
        df_item_ratings = self.get_df_item_ratings_from(item_id)
        return df_item_ratings['rating'].tolist()

    def get_preference_value(self, user_id, item_id):

        """
        Gets preference value from the given user for the given item or null if none exists.
        :param user_id: id of the user.
        :param item_id: id of the item.
        :return: preference value from the given user for the given item.
        """
        df_user_ratings = self.get_df_user_ratings_from(user_id)
        return float(df_user_ratings[df_user_ratings['item_id'] == item_id]['rating'])

    def get_preference_time(self, user_id, item_id):

        """
        Gets time at which preference was set or null if no preference exists or its time is not known.
        :param user_id: id of the user.
        :param item_id: id of the item.
        :return: preference time value from the given user for the given item.
        """

        df_user_ratings = self.get_df_user_ratings_from(user_id)

        row = df_user_ratings[df_user_ratings['item_id'] == item_id]
        preference = float(row['rating'])
        timestamp = row['timestamp'].tolist()
        return preference, timestamp[0]

    def get_num_users_with_preference_for(self, item_id):

        """
        Gets the number of users who have expressed a preference for the item.
        :param item_id: id of the item.
        :return: number of users.
        """
        df_item_ratings = self.df_ratings[self.df_ratings['item_id'] == item_id]
        return len(df_item_ratings.index)

    def get_num_users_with_preference_for_ids(self, item_id1, item_id2):

        """
        Gets the number of users who have expressed a preference for the items.
        :param item_id1: id of the item.
        :param item_id2: id of the item.
        :return: number of users.
        """
        df_item_ratings = self.df_ratings.loc[self.df_ratings['item_id'].isin([item_id1, item_id2])]
        df_duplicated_users = df_item_ratings.duplicated(['user_id'])
        return sum(list(df_duplicated_users))

    def get_max_preference(self):

        """
        Gets the maximum preference value that is possible in the current problem domain being evaluated.
        For example, if the domain is movie ratings on a scale of 1 to 5, this should be 5.
        """
        return float(self.df_ratings.loc[self.df_ratings['rating'].idxmax()]['rating'])

    def get_min_preference(self):

        """
        Gets the minimum preference value that is possible in the current problem domain being evaluated.
        For example, if the domain is movie ratings on a scale of 1 to 5, this should be 1.
        """
        return float(self.df_ratings.loc[self.df_ratings['rating'].idxmin()]['rating'])


    def get_num_users(self):

        """
        Gets total number of users known to the input_model.
        """
        return len(self.df_ratings['user_id'].unique())

    def get_user_ids(self):

        """
        Gets users known to the input_model.
        """
        return self.df_ratings['user_id'].unique()

    def get_num_items(self):

        """
        Gets total number of items rated by users. It does not have to coincide with the number of total items.
        """
        return len(self.df_ratings['item_id'].unique())

    def get_item_ids(self):

        """
        Gets items rated by users. It does not have to coincide with the number of total items.
        """

        return self.df_ratings['item_id'].unique()

    def get_num_ratings(self):

        """
        Gets total number of ratings known to the input_model.
        """
        return len(self.df_ratings.axes[0])

    def get_user_with_most_ratings(self):

        """
        Gets the user with the most ratings.
        """
        return self.df_ratings['user_id'].value_counts().idxmax()

    def get_user_with_fewer_ratings(self):

        """
        Gets the user with fewer ratings.
        """
        return self.df_ratings['user_id'].value_counts().idxmin()

    def get_item_with_most_ratings(self):

        """
        Gets the item with the most ratings: popular items.
        """
        return self.df_ratings['item_id'].value_counts().idxmax()

    def get_item_with_fewer_ratings(self):

        """
        Gets the item with fewer ratings: rare items.
        """
        return self.df_ratings['item_id'].value_counts().idxmin()

    def get_average_rating_from_user(self, user_id):

        """
        Gets the average rating for a user.
        :param user_id: id of the user.
        """

        avg = sum(self.get_preferences_from_user(user_id)) / len(self.get_preferences_from_user(user_id))
        return round(avg, 2)

    def get_items_per_user(self):

        """
        Gets the items seen per user.
        """
        items_per_user = dict()
        for i in range(1, self.get_num_users() + 1):
            items = len(self.get_item_ids_from_user(i))
            items_per_user[i] = items
        return items_per_user

    def get_average_num_items(self):

        """
        Get the average number of items.
        """

        average = 0.0
        n = self.get_num_users()
        for i in range(1, n + 1):
            average+= len(self.get_item_ids_from_user(i))
        return(average/n)


def main():
    ratings_file_path = 'C:/Users/ssabroso/Documents/GitLab/pct_2019/Semantic_aware_models/resources/ml-1m_DBpedia/model/input_model_1_5/train_bin_verified_sep_0.csv'
    ratings_data_model = RatingDataModel(ratings_file_path= ratings_file_path, separator='	')

    items = ratings_data_model.get_items_per_user()
    print('Items del usuario 4169: ' , items[4169])
    print('Items del usuario 623: ', items[623])
    print('Items del usuario 3020: ', items[3020])
    print('Items del usuario 3234: ', items[3234])

if __name__ == '__main__':
    main()



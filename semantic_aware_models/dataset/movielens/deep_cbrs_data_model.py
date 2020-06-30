from semantic_aware_models.dataset.movielens.movielens_data_model import ItemUnstructuredDataModel
from semantic_aware_models.dataset.movielens.movielens_data_model import ItemStructuredDataModel
import torch


class DeepCBRSDataModel:

    def __init__(self):
        pass

    # Reads the text descriptions associated to each item from a given path.
    # Parameters:
    #         train_path: folder in which item descriptions are stored
    # Output:
    #         dictionary structure which contains the following data:
    #         items: item descriptions
    #         item2pos: dictionary which maps item ids to position in the dataset
    #         pos2item: dictionary which maps position in the dataset to item ids
    #         token2id: dictionary which maps tokens to word identifiers
    #         max_item_len: maximum number of words in a text description
    def read_items_data(self, train_path):
        items = dict()
        item2pos = dict()
        pos2item = dict()
        token2id = dict()
        num_items = 1
        num_tokens = 1
        max_item_len = 0

        items_unstructured_data_model = ItemUnstructuredDataModel(train_path, separator='::')
        item_ids = items_unstructured_data_model.get_item_ids()
        descriptions = items_unstructured_data_model.get_description()

        for idx, description in enumerate(descriptions):
            item_id = item_ids[idx]
            item2pos[item_id] = num_items
            pos2item[num_items] = item_id

            words = description.split(' ')
            # print('words: ', words)
            item_words = list()
            for word in words:
                # print('word: ', word)
                if word not in token2id:
                   token2id[word] = num_tokens
                   num_tokens += 1

                item_words.append(token2id[word])
            # print('item_words: ', item_words)

            len_words = len(item_words)
            if len_words > max_item_len:
                max_item_len = len_words

            items[item_id] = item_words
            num_items += 1

        return {'items': items, 'item2pos': item2pos, 'pos2item': pos2item, 'token2id': token2id, 'max_item_len': max_item_len}

    # Pads item description according to the maximum number of tokens in the item descriptions.
    def pad_items_data(self, items_data):
        data = torch.zeros(int(len(items_data['items']) + 1), int(items_data['max_item_len']), dtype=torch.int64)  # :zero()
        output = dict()

        for item_id, tokens in items_data['items'].items():
            for i, token in enumerate(tokens):
                data[items_data['item2pos'][item_id]][i] = token
                output[item_id]= data[items_data['item2pos'][item_id]]
        return output

    # Loads genres metadata associated to each item.
    #    Parameters:
    #     - genres_filename: name of the file containing genres information for each item in JSON format
    #     - item2pos: maps item ids to item position in the dataset
    def load_items_genres(self, genres_filename, item2pos):
        genre2id = dict()
        id2genre = dict()
        genres = dict()
        item2pos = dict()
        pos2item = dict()
        num_genres = 1
        max_num_genres = 0

        items_structured_data_model = ItemStructuredDataModel(genres_filename, separator='::')
        item_ids = items_structured_data_model.get_item_ids()
        data = items_structured_data_model.get_genres()
        # print('data: ', data)

        for idx, item_genres_str in enumerate(data):
            item_genres = str(item_genres_str).split('|')
            item_id = item_ids[idx]
            item2pos[item_id] = idx+1
            pos2item[idx+1] = item_id

            if item_id:
                item = int(item_id)
                item_mapped_genres = list()

                len_item_genres = len(item_genres)
                if len_item_genres > max_num_genres:
                    max_num_genres = len_item_genres

                for item_genre in item_genres:
                    if item_genre not in genre2id:
                        genre2id[item_genre] = num_genres
                        id2genre[num_genres] = item_genre
                        num_genres += 1

                    item_mapped_genres.append(genre2id[item_genre])

                genres[item] = item_mapped_genres

        return {'genres': genres, 'genre2id': genre2id, 'id2genre': id2genre, 'pos2item': pos2item, 'item2pos': item2pos, 'max_num_genres': max_num_genres}

    # Pads item genres according to the maximum number of genres associated to each item
    def pad_genres_data(self, genres_data):
        non_retrieved_genres = 3
        data = torch.zeros(len(genres_data['genres']) + non_retrieved_genres, genres_data['max_num_genres'], dtype=torch.int64)  # :zero()
        output = dict()

        # print('genres_data[genres]: ', genres_data['genres'])
        for item_pos, genres in genres_data['genres'].items():
            # print('genres: ', genres)
            for i, genre in enumerate(genres):
                data[genres_data['item2pos'][item_pos]][i] = genre
                output[item_pos] = data[genres_data['item2pos'][item_pos]]

        return output

    # Loads authors metadata associated to each item.
    #    Parameters:
    #     - authors_filename: name of the file containing authors information for each item in JSON format
    #     - item2pos: maps item ids to item position in the dataset
    def load_items_authors(self, authors_filename, item2pos):
        author2id = dict()
        id2author = dict()
        authors = dict()
        item2pos = dict()
        pos2item = dict()
        num_authors = 1
        max_num_authors = 0

        items_structured_data_model = ItemStructuredDataModel(authors_filename, separator='::')
        item_ids = items_structured_data_model.get_item_ids()
        data = items_structured_data_model.get_starring()
        #print('data: ', data)

        for idx, item_authors_str in enumerate(data):
            item_authors = str(item_authors_str).split('|')
            item_id = item_ids[idx]
            item2pos[item_id] = idx + 1
            pos2item[idx + 1] = item_id

            if item_id:
                item = int(item_id)
                item_mapped_authors = list()

                len_item_authors = len(item_authors)
                if len_item_authors > max_num_authors:
                    max_num_authors = len_item_authors

                for item_author in item_authors:
                    if item_author not in author2id:
                        author2id[item_author] = num_authors
                        id2author[num_authors] = item_author
                        num_authors += 1

                    item_mapped_authors.append(author2id[item_author])

                authors[item] = item_mapped_authors

        return {'authors': authors, 'author2id': author2id, 'id2author': id2author, 'pos2item': pos2item, 'item2pos': item2pos, 'max_num_authors': max_num_authors}

    # Pads item authors according to the maximum number of authors associated to each item
    def pad_authors_data(self, authors_data):
        non_retrieved_authors = 359
        data = torch.zeros(len(authors_data['authors']) + non_retrieved_authors, authors_data['max_num_authors'], dtype=torch.int64)  # :zero()
        output = dict()

        for item_pos, authors in authors_data['authors'].items():
            for i, author in enumerate(authors):
                data[authors_data['item2pos'][item_pos]][i] = author
                output[item_pos] = data[authors_data['item2pos'][item_pos]]

        return output

    # Loads directors metadata associated to each item.
    #    Parameters:
    #     - directors_filename: name of the file containing directors information for each item in JSON format
    #     - item2pos: maps item ids to item position in the dataset
    def load_items_directors(self, directors_filename, item2pos):
        director2id = dict()
        id2director = dict()
        directors = dict()
        item2pos = dict()
        pos2item = dict()
        num_directors = 1
        max_num_directors = 0

        items_structured_data_model = ItemStructuredDataModel(directors_filename, separator='::')
        item_ids = items_structured_data_model.get_item_ids()
        data = items_structured_data_model.get_director()
        #print('data: ', data)

        for idx, item_directors_str in enumerate(data):
            item_directors = str(item_directors_str).split('|')
            item_id = item_ids[idx]
            item2pos[item_id] = idx + 1
            pos2item[idx + 1] = item_id

            if item_id:
                item = int(item_id)
                item_mapped_directors = list()

                len_item_directors = len(item_directors)
                if len_item_directors > max_num_directors:
                    max_num_directors = len_item_directors

                for item_director in item_directors:
                    if item_director not in director2id:
                        director2id[item_director] = num_directors
                        id2director[num_directors] = item_director
                        num_directors += 1

                    item_mapped_directors.append(director2id[item_director])

                directors[item] = item_mapped_directors

        return {'directors': directors, 'director2id': director2id, 'id2director': id2director, 'pos2item': pos2item, 'item2pos': item2pos, 'max_num_directors': max_num_directors}

    # Pads item directors according to the maximum number of directors associated to each item
    def pad_directors_data(self, directors_data):
        non_retrieved_directors = 359
        data = torch.zeros(len(directors_data['directors']) + non_retrieved_directors, directors_data['max_num_directors'], dtype=torch.int64)  # :zero()
        output=dict()

        for item_pos, directors in directors_data["directors"].items():
            for i, director in enumerate(directors):
                data[directors_data['item2pos'][item_pos]][i] = director
                output[item_pos] = data[directors_data['item2pos'][item_pos]]

        return output

    # Loads wiki categories metadata associated to each item.
    #    Parameters:
    #     - wiki_categories_filename: name of the file containing wiki categories information for each item in JSON format
    #     - item2pos: maps item ids to item position in the dataset
    def load_items_wiki_categories(self, wiki_categories_filename, item2pos):
        wiki_category2id = dict()
        id2wiki_category = dict()
        wiki_categories = dict()
        item2pos = dict()
        pos2item = dict()
        num_wiki_categories = 1
        max_num_wiki_categories = 0

        items_structured_data_model = ItemStructuredDataModel(wiki_categories_filename, separator='::')
        item_ids = items_structured_data_model.get_item_ids()
        data = items_structured_data_model.get_subject()
        #print('data: ', data)

        for idx, item_wiki_categories_str in enumerate(data):
            item_wiki_categories = str(item_wiki_categories_str).split('|')
            item_id = item_ids[idx]
            item2pos[item_id] = idx + 1
            pos2item[idx + 1] = item_id

            if item_id:
               item = int(item_id)
               item_mapped_wiki_categories =list()

               len_item_wiki_categories = len(item_wiki_categories)
               if len_item_wiki_categories > max_num_wiki_categories:
                   max_num_wiki_categories = len_item_wiki_categories

               for item_wiki_category in item_wiki_categories:
                    if item_wiki_category not in wiki_category2id:
                        wiki_category2id[item_wiki_category] = num_wiki_categories
                        id2wiki_category[num_wiki_categories] = item_wiki_category
                        num_wiki_categories += 1

                    item_mapped_wiki_categories.append(wiki_category2id[item_wiki_category])

               wiki_categories[item] = item_mapped_wiki_categories

        return {'wiki_categories': wiki_categories, 'wiki_category2id': wiki_category2id, 'id2wiki_category': id2wiki_category, 'pos2item': pos2item, 'item2pos': item2pos,
                                        'max_num_wiki_categories': max_num_wiki_categories}

    # Pads item wiki categories according to the maximum number of wiki categories associated to each item
    def pad_wiki_categories_data(self, wiki_categories_data):
        non_retrieved_wiki_categories = 359
        data = torch.zeros(len(wiki_categories_data['wiki_categories']) + non_retrieved_wiki_categories, wiki_categories_data['max_num_wiki_categories'], dtype=torch.int64)  # :zero()
        output = dict()

        for item_pos, wiki_categories in wiki_categories_data['wiki_categories'].items():
            for i, wiki_category in enumerate(wiki_categories):
                data[wiki_categories_data['item2pos'][item_pos]][i] = wiki_category
                output[item_pos] = data[wiki_categories_data['item2pos'][item_pos]]

        return output

    # Loads properties metadata associated to each item.
    #    Parameters:
    #     - properties_filename: name of the file containing properties information for each item in JSON format
    #     - item2pos: maps item ids to item position in the dataset
    def load_items_properties(self, properties_filename, item2pos):
        property2id = dict()
        id2property = dict()
        properties = dict()
        item2pos = dict()
        pos2item = dict()

        num_properties = 1
        max_num_properties = 0

        items_structured_data_model = ItemStructuredDataModel(properties_filename, separator='::')
        data = list()  # TODO ¿?
        item_ids = items_structured_data_model.get_item_ids()

        for idx, item_properties_str in enumerate(data):
            item_properties = str(item_properties_str).split('|')
            item_id = item_ids[idx]

            if item_id:
                item = int(item_id)
                item_mapped_properties = list()
                item2pos[item_id] = idx + 1
                pos2item[idx + 1] = item_id

                len_item_properties = len(item_properties)
                if len_item_properties > max_num_properties:
                    max_num_properties = len_item_properties

                for item_property in item_properties:
                    if item_property not in property2id:
                        property2id[item_property] = num_properties
                        id2property[num_properties] = item_property
                        num_properties += 1

                    item_mapped_properties.append(property2id[item_property])

        return {'properties': properties,'property2id': property2id,'id2property': id2property, 'pos2item': pos2item, 'item2pos': item2pos, 'max_num_properties': max_num_properties}

    # Pads item properties according to the maximum number of properties associated to each item:
    def pad_properties_data(self, properties_data):
        non_retrieved_properties = 359
        data = torch.zeros(len(properties_data['properties']) + non_retrieved_properties, properties_data['max_num_properties'], dtype=torch.int64)  # :zero()
        output=dict()

        for item_pos, properties in properties_data['properties'].items():
            for i, property in enumerate(properties):
                data[properties_data['item2pos'][item_pos]][i] = property
                output[item_pos] = data[properties_data['item2pos'][item_pos]]

        return output

import os
import torch
import torch.nn as nn
import torch.optim as optim
from semantic_aware_models.utils.gpu import GPU
from semantic_aware_models.models.classification.brnn_classifier import DeepCBRS
from semantic_aware_models.dataset.movielens.deep_cbrs_data_model import DeepCBRSDataModel
from semantic_aware_models.dataset.movielens.movielens_data_model import RatingDataModel


def main():
    """
        Experiment main entry point for the classifier used by DeepCBRS
    """
    gpu = GPU()
    device = gpu.get_default_device()

    data_model = DeepCBRSDataModel()

    path = '<main_resources_path>'
    conf_data = {'items': os.path.join(path, '<path of items descriptions>'),
                 'genres': os.path.join(path, '<filename of items genres>'),
                 'authors': os.path.join(path, '<filename of items authors>'),
                 'directors': os.path.join(path, '<filename of items directors>'),
                 'properties': os.path.join(path, '<filename of items directors>'),
                 'wiki_categories': os.path.join(path, '<filename of items categories>'),
                 'models_mapping': os.path.join(path, ''),  # dictionary which associates input_test files to models
                 'predictions': os.path.join(path, ''),  # generated predictions filename
                 'batch_size': os.path.join(path, ''),  # number of examples in a batch
                 'topn': os.path.join(path, '')  # list of cutoff values
                 }

    # ITEMS DESCRIPTIONS:
    # Loading items data:
    items_data = data_model.read_items_data(train_path=conf_data['items'])
    # Padding items data:
    print("Max len seq: ", items_data['max_item_len'])
    items_data['items'] = data_model.pad_items_data(items_data)

    # GENRES:
    # Loading genres data:
    genres_data = data_model.load_items_genres(genres_filename=conf_data['genres'], item2pos=items_data['item2pos'])
    # Padding genres data:s
    # print(genres_data)
    genres_data['genres'] = data_model.pad_genres_data(genres_data)

    # AUTHORS:
    # Loading authors data:
    authors_data = data_model.load_items_authors(authors_filename=conf_data['authors'], item2pos=items_data['item2pos'])
    # Padding authors data:
    authors_data['authors'] = data_model.pad_authors_data(authors_data)

    # DIRECTORS:
    # Loading directors data:
    directors_data = data_model.load_items_directors(directors_filename=conf_data['directors'],
                                                     item2pos=items_data['item2pos'])
    # print('directors_data: ', directors_data)
    # Padding directors data:
    directors_data['directors'] = data_model.pad_directors_data(directors_data)

    # WIKI_CATEGORIES:
    # Load wiki_categories data:
    wiki_categories_data = data_model.load_items_wiki_categories(wiki_categories_filename=conf_data['wiki_categories'],
                                                                 item2pos=items_data['item2pos'])
    # Padding wiki_categories:
    wiki_categories_data['wiki_categories'] = data_model.pad_wiki_categories_data(wiki_categories_data)

    # RATINGS
    train_ratings_path = os.path.join(path, '<train_ratings_path>')
    ratings_data = RatingDataModel(ratings_file_path=train_ratings_path, separator='	')

    ##########################################################################################
    # DEEPCBRS TEST:
    # We have already load the data from all movies to create the vocabularies.
    # Fix an user, for instance the user 1
    user_id = 1
    batch_size = 2  # 32
    num_epochs = 5  # 20

    # Create the DeepCBRS model:
    model = DeepCBRS(items_data=items_data, genres_data=genres_data, authors_data=authors_data,
                     directors_data=directors_data, batch_size=batch_size, wiki_categories_data=wiki_categories_data)

    model = gpu.to_device(model, device)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    # Get the ids of the movies seen by the user fixed:
    list_ids = ratings_data.get_item_ids_from_user(user_id=user_id)
    indices = torch.tensor(list_ids)
    samples = list(indices.split(batch_size))
    samples.pop()

    for epoch in range(num_epochs):
        for i in range(len(samples)):
            ids_samples = list()
            # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into. Tensors of word indices.
            for j in range(len(samples[i])):
                ids_samples.append(samples[i][j].item())
            item_tensors = tuple([items_data['items'][x] for x in ids_samples])
            items_input = (torch.cat(item_tensors, 0)).view(batch_size, -1)
            items_input = gpu.to_device(items_input, device)

            genres_tensors = tuple([genres_data['genres'][x] for x in ids_samples])
            genres_input = (torch.cat(genres_tensors, 0)).view(batch_size, -1)
            genres_input = gpu.to_device(genres_input, device)

            authors_tensors = tuple([authors_data['authors'][x] for x in ids_samples])
            authors_input = (torch.cat(authors_tensors, 0)).view(batch_size, -1)
            authors_input = gpu.to_device(authors_input, device)

            directors_tensors = tuple([directors_data['directors'][x] for x in ids_samples])
            directors_input = (torch.cat(directors_tensors, 0)).view(batch_size, -1)
            directors_input = gpu.to_device(directors_input, device)

            wiki_categories_tensors = tuple([wiki_categories_data['wiki_categories'][x] for x in ids_samples])
            wiki_categories_input = (torch.cat(wiki_categories_tensors, 0)).view(batch_size, -1)
            wiki_categories_input = gpu.to_device(wiki_categories_input, device)

            list_ratings = [ratings_data.get_preference_value(user_id=user_id, item_id=x) for x in ids_samples]
            ratings = torch.tensor(list_ratings).view(batch_size, -1)
            ratings = gpu.to_device(ratings, device)

            # Step 3. Run our forward pass.
            predictions = model.forward(items_input, genres_input, authors_input, directors_input, wiki_categories_input)

            # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            print(predictions)
            loss = loss_function(predictions, ratings)
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()

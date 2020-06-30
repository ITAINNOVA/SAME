import torch
import torch.nn as nn
import torch.nn.parallel
from semantic_aware_models.utils.gpu import GPU

gpu = GPU()


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(gpu.get_default_device())
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Initialize parameters
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        h0 = gpu.to_device(h0, gpu.get_default_device())
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = gpu.to_device(c0, gpu.get_default_device())

        # Forward propagate LSTM
        out, hidden = self.lstm(x, (h0, c0))
        out = gpu.to_device(out, gpu.get_default_device())
        hidden = gpu.to_device(hidden, gpu.get_default_device())

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class UnstructuredModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size):
        super(UnstructuredModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(vocab_size + 1, embedding_dim).to(gpu.get_default_device())

        biRNN = BiRNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, num_classes=tagset_size)

        self.brnn = biRNN.to(gpu.get_default_device())
        self.linear_1 = nn.Linear(int(embedding_dim), int(embedding_dim / 2))
        self.linear_2 = nn.Linear(int(embedding_dim / 2), int(embedding_dim))

    def forward(self, input):
        embeds = self.word_embeddings(input)

        brnn_out = self.brnn(embeds.view(-1, self.batch_size, self.embedding_dim))
        brnn_out = brnn_out.view(self.batch_size, -1, self.embedding_dim)
        aux = brnn_out.mean(-2)

        tag_space = torch.tanh(self.linear_1(aux))
        tag_scores = torch.tanh(self.linear_2(tag_space))
        return tag_scores


class StructuredModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(StructuredModel, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size + 1, embedding_dim).to(gpu.get_default_device())
        self.linear_1 = nn.Linear(int(embedding_dim), int(embedding_dim / 2))
        self.linear_2 = nn.Linear(int(embedding_dim / 2), int(embedding_dim))

    def forward(self, input):
        embeds = self.word_embeddings(input)
        aux = embeds.mean(-2)
        tag_space = torch.tanh(self.linear_1(aux))
        tag_scores = torch.tanh(self.linear_2(tag_space))
        return tag_scores


class DeepCBRS(nn.Module):
    """
        This implementation is loosely based on their LUA's homonym published at https://github.com/nlp-deepcbrs/amar
    """
    def __init__(self, items_data, genres_data, authors_data,
                 directors_data, wiki_categories_data, batch_size):  # ratings_data, initial_weights, batch_size
        super(DeepCBRS, self).__init__()

        self.item_embeddings_size = 50
        self.genre_embeddings_size = 50
        self.author_embeddings_size = 50
        self.director_embeddings_size = 50
        self.property_embeddings_size = 50
        self.wiki_category_embeddings_size = 50
        self.half_embeddings_size = self.item_embeddings_size / 2
        self.hidden_dense_layer_size = self.item_embeddings_size
        self.batch_size = batch_size

        self.num_tokens = len(items_data['token2id'])
        self.genres_vocab_size = len(genres_data['genre2id'])
        self.authors_vocab_size = len(authors_data['author2id'])
        self.directors_vocab_size = len(directors_data['director2id'])
        self.wiki_categorires_vocab_size = len(wiki_categories_data['wiki_category2id'])

        # Networks building
        self.items_net = UnstructuredModel(embedding_dim=self.item_embeddings_size,
                                           hidden_dim=self.hidden_dense_layer_size,
                                           vocab_size=self.num_tokens, tagset_size=self.item_embeddings_size,
                                           batch_size=self.batch_size)
        if genres_data:
            self.hidden_dense_layer_size = self.hidden_dense_layer_size + self.genre_embeddings_size
            self.genres_net = StructuredModel(embedding_dim=self.genre_embeddings_size,
                                              vocab_size=self.genres_vocab_size)

        if authors_data:
            self.hidden_dense_layer_size = self.hidden_dense_layer_size + self.author_embeddings_size
            self.authors_net = StructuredModel(embedding_dim=self.author_embeddings_size,
                                               vocab_size=self.authors_vocab_size)

        if directors_data:
            self.hidden_dense_layer_size = self.hidden_dense_layer_size + self.director_embeddings_size
            self.directors_net = StructuredModel(embedding_dim=self.director_embeddings_size,
                                                 vocab_size=self.directors_vocab_size)

        if wiki_categories_data:
            self.hidden_dense_layer_size = self.hidden_dense_layer_size + self.wiki_category_embeddings_size
            self.wiki_categories_net = StructuredModel(embedding_dim=self.wiki_category_embeddings_size,
                                                 vocab_size=self.wiki_categorires_vocab_size)

        self.half_embeddings_size = self.hidden_dense_layer_size / 2
        self.linear_1 = nn.Linear(self.hidden_dense_layer_size, 5)

    def forward(self, item_input, genres_input, authors_input, directors_input, wiki_categories_input):
        item_output = self.items_net(item_input)
        genres_output = self.genres_net(genres_input)
        authors_output = self.authors_net(authors_input)
        directors_output = self.authors_net(directors_input)
        wiki_categories_output = self.wiki_categories_net(wiki_categories_input)

        join_table = torch.cat((item_output, genres_output, authors_output, directors_output, wiki_categories_output), -1)
        output = torch.sigmoid(self.linear_1(join_table))

        return output

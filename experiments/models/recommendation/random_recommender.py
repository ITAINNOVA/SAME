from semantic_aware_models.models.recommendation.random_recommender import RandomRecommender
from surprise.reader import Reader
import json


def main():
    with open('random_recommender_config.json', 'r') as f:
        config = json.load(f)

        path = config['path']
        separator = config['separator']
        n_folds = config['n_folds']


    output_recommendation_file_path = path + '<output_recommendation_file_path>'
    input_file_path = path + '<input_file_path>'
    ratings_file_path = path + '<ratings_file_path>'
    random_path = output_recommendation_file_path + 'random/'
    reader = Reader(line_format='user item rating timestamp', sep='	')

    recommender = RandomRecommender(ratings_file_path=ratings_file_path, separator=separator)
    recommender.recommend_rival(n_folds=n_folds, train_test_file_path=input_file_path, reader=reader, recommendation_file_path=random_path)


if __name__ == '__main__':
    main()

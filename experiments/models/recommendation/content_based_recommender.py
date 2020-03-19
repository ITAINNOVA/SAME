from semantic_aware_models.models.recommendation.content_based_recommender import ContentBasedRecommender
import json


def main():
    with open('content_based_recommender_config.json', 'r') as f:
        config = json.load(f)

        path = config['path']
        separator = config['separator']
        n_folds = config['n_folds']


    items_unstructured_file_path = path + '<items_unstructured_file_path>'
    train_test_file_path = path + '<train_test_file_path>'
    recommendation_file_path = path + '<recommendation_file_path>'

    cbrs = ContentBasedRecommender(separator, items_file_path=items_unstructured_file_path)
    cbrs.recommend_rival(n_folds, train_test_file_path, recommendation_file_path)


if __name__ == '__main__':
    main()
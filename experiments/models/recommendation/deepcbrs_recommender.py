from semantic_aware_models.models.recommendation.deepcbrs_recommender import *
import os
import torch


def main():
    path = '<main_resources_path>'

    models_path = os.path.join(path, '<model_path/>')
    ratings_file_path = os.path.join(path, '<ratings_file_path>')
    unstructured_file_path = os.path.join(path, '<unstructured_file_path>')
    structured_file_path = os.path.join(path, '<structured_file_path>')

    recommnender = DeepCBRSRecommender(separator='	', ratings_file_path=ratings_file_path,
                                       structured_file_path=structured_file_path, unstructured_file_path=unstructured_file_path,
                                       models_path=models_path)

    print('Using: ', recommnender.device)
    print('Available: ', torch.cuda.device_count())


    train_test_file_path = os.path.join(path, '<train_test_file_path>')

    recommendation_file_path = os.path.join(path, '<output_recommendation_file_path>')

    recommnender.recommend_rival(n_folds=1, train_test_file_path=train_test_file_path, recommendation_file_path=recommendation_file_path)


if __name__ == '__main__':
    main()

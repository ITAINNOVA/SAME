from semantic_aware_models.models.recommendation.bert_recommender import *
import torch
import os
import sys
import logging
import json


def main():
    if len(sys.argv) == 1:
        config_path = "./bert_recommender_config.json"
    else:
        config_path = sys.argv[1]

    print("Loading configuration: " + config_path)

    ### ARGUMENTS
    with open(config_path, 'r') as f:
        config = json.load(f)

        if config['local_rank'] is not None:
            local_rank = config['local_rank']
        else:
            local_rank = -1
        if config['no_cuda'] is not None:
            no_cuda = config['no_cuda']
        else:
            no_cuda = True

        if config['no_cuda'] is not None:
            no_cuda = config['no_cuda']
        else:
            no_cuda = True
        # To add
        fp16 = False
        seed = 42

        output_dir = config['output_dir']
        task_name = config['task_name']
        path = config['path']

    ### Set cuda environment
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    # if os.path.exists(output_dir) and os.listdir(output_dir):
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the recommender

    descriptions_path = os.path.join(path, '<descriptions_path>')
    ratings_file_path = os.path.join(path, '<ratings_file_path>')


    bert_recommender = BertRecommender(items_file_path=descriptions_path, ratings_file_path=ratings_file_path, config_model=config)

    # Rival Recommend

    train_test_file_path = os.path.join(path, '<train_test_file_path>')

    recommendation_file_path = os.path.join(path, '<recommendation_file_path>')

    bert_recommender.recommend_rival(n_folds=5, train_test_file_path=train_test_file_path, recommendation_file_path=recommendation_file_path)



if __name__ == '__main__':
    main()

import sys
import torch
import json
import logging
import os

from semantic_aware_models.models.classification.bert_classifier import BertClassifier
from semantic_aware_models.models.language.bert.data_processors import processors

from semantic_aware_models.dataset.movielens.movielens_data_model import RatingDataModel, ItemUnstructuredDataModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename='bert_classifier.log', filemode='w')


def main():
    """
        Experiment main entry point (adaption from HuggingFace Transformers)
    """
    if len(sys.argv) == 1:
        config_path = "./bert_classifier_config.json"
    else:
        config_path = sys.argv[1]

    print("Loading configuration: " + config_path)


    # ARGUMENTS
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

        task_name = config['task_name']
        path = config['path']
        output_dir = os.path.join(path, config['output_dir'])

    # Set cuda environment
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    # if os.path.exists(output_dir) and os.listdir(output_dir):
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TRAIN
    bert_cbrs = BertClassifier()
    user_id = 4
    processor = processors[task_name]()

    ratings_train_path = os.path.join(path, 'resources/...' )
    descriptions_path = os.path.join(path, 'resources/...')

    descriptions = ItemUnstructuredDataModel(descriptions_path, separator='::')
    ratings_train_data_model = RatingDataModel(ratings_file_path=ratings_train_path, separator="	")

    train_data = processor.get_train_examples(descriptions=descriptions, ratings=ratings_train_data_model,
                                                 user_id=user_id)

    bert_cbrs.train_model(config=config, train_data=train_data)

    # TEST
    ratings_test_path = os.path.join(path, 'resources/...')

    ratings_test_data_model = RatingDataModel(ratings_file_path=ratings_test_path, separator="	")

    test_data = processor.get_train_examples(descriptions=descriptions, ratings=ratings_test_data_model,
                                                 user_id=user_id)

    bert_cbrs.test_model(config=config, test_data=test_data)


if __name__ == "__main__":
    main()

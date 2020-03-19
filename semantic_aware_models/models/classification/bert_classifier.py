import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME

from semantic_aware_models.models.language.bert.data_processors import processors, output_modes
from semantic_aware_models.models.language.bert.data_processors import convert_examples_to_features
from semantic_aware_models.models.classification.abstract_classifier import AbstractClassifier


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename='bert_classifier.log', filemode='w')


class BertClassifier(AbstractClassifier):
    """
        This implementation fine-tunes given BERT model in order to perform simple (one-class) classification.
    """

    def __init__(self):
        """
        Create a Bert Classifier object.
        """
        super(AbstractClassifier, self).__init__()

    def train_model(self, config, train_data):

        """
        BERT classifier training.
        :param config: Bert model configuration in a JSON file.
        :param train_data: Dataset for bert model training, composed by the information and label of each item.
        :return: The Bert classifier model trained with the train data.
        """

        train_batch_size = config['train_batch_size']
        gradient_accumulation_steps = config['gradient_accumulation_steps']
        num_train_epochs = config['num_train_epochs']
        local_rank = config['local_rank']
        bert_model = config['bert_model']
        learning_rate = config['learning_rate']
        warmup_proportion = config['warmup_proportion']
        task_name = config['task_name']
        no_cuda = config['no_cuda']
        output_dir = config['output_dir']

        do_lower_case = config['do_lower_case']
        max_seq_length = config['max_seq_length']

        train_batch_size = train_batch_size // gradient_accumulation_steps

        # Task utils
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        processor = processors[task_name]()
        output_mode = output_modes[task_name]

        label_list = processor.get_labels()
        num_labels = len(label_list)

        # BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        num_train_optimization_steps = None

        # Using GPU
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

        num_train_optimization_steps = int(
            len(train_data) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
        if local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare the model
        model = BertForSequenceClassification.from_pretrained(bert_model,
                                                              num_labels=num_labels)

        model.to(device)
        if local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare Optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

        # Training
        global_step = 0
        tr_loss = 0
        train_features, _ = convert_examples_to_features(train_data, label_list, max_seq_length, tokenizer,
                                                         output_mode)
        # print("***** Running training *****")
        # print("  Num examples = %d", len(train_data))
        # print("  Batch size = %d", train_batch_size)
        # print("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        model.train()
        for i_ in range(int(num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)


                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # print('Loss after epoch {}'.format(tr_loss / nb_tr_steps))
                # print('Eval after epoch {}'.format(i_ + 1))

        # Save the model
        if (local_rank == -1 or torch.distributed.get_rank() == 0):

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`

            output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir)

        model.to(device)

        return model

    def test_model(self, config, test_data):

        """
        BERT classifier testing.
        :param config: Bert model configuration in a JSON file.
        :param test_data: Dataset for bert model testing, composed by the information and label of each item.
        :return: Predictions of the Bert Model trained with train_model method.
        """

        test_batch_size = config['test_batch_size']
        local_rank = config['local_rank']
        task_name = config['task_name']
        no_cuda = config['no_cuda']
        output_dir = config['output_dir']
        do_lower_case = config['do_lower_case']
        max_seq_length = config['max_seq_length']



        ### Task utils
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        processor = processors[task_name]()
        output_mode = output_modes[task_name]

        label_list = processor.get_labels()
        num_labels = len(label_list)

        # BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)

        # Using GPU
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

        model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)

        model.to(device)

        if local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)


        test_features, id2label = convert_examples_to_features(test_data, label_list, max_seq_length, tokenizer,
                                                         "classification")
        # print("***** Running input_test *****")
        # print("  Num examples = %d", len(test_data))
        # print("  Batch size = %d", test_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        model.eval()
        preds = []

        for step, batch in enumerate(test_dataloader):
            input_ids, input_mask, segment_ids = batch

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:

                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)


        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        return preds

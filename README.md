# SAME: Semantic Aware ModeEls

SAME: Semantic Aware ModeEls is a framework with different traditional recommendation models and a semantic-aware content-based recommendation model that exploits textual features of items obtained from the Linked Open Data and deep learning transformers like BERT (Bidirectional Encoder Representations from Transformers).

## Description

In this project, we have included the following traditional recommendation models:

- A random recommendation model that predicts a random rating based on the distribution of the training set, which is assumed to be normal. This model was provided by Surprise library.

- A traditional content-based recommendation model based on Vector Space Model (VSM) to represent the textual information of items, by using TF-IDF weights. This model was implemented by using Sklearn library.

- A semantic-aware content-based recommendation model, based on BERT classifier able to train and test any text information with its related labels.

- An alternative of the deep content-based recommendation model proposed by Musto, called [deepCBRS](https://github.com/nlp-deepcbrs/amar), that uses a binary classifier based on Bidirectional Recurrent Neuronal Networks (BRNNs). Ir order to use 5 classes, we adapted the source code of the original model. In addition, we do not use embedding models to represent the textual information of the items.

C. Musto, T. Franza, G. Semeraro, M. de Gemmis, and P. Lops, “Deep content-based recommender systems exploiting Recurrent Neural Networks and Linked Open Data,” in 26th Conference on User Modeling, Adaptation and Personalization (UMAP). ACM, July 2018, pp. 239–244.

## Requirements

The libraries used in this project with its respective versions can be seen in `requirements.txt`.

## Usage

1. To use the BERT recommendation model, prepare the task and the input correctly. Some examples are available in `data_processors.py`.
2. Create JSON configuration file for each recommender.
3. Executable scripts are provided under the `/experiments` directory. Specifically, recommendation scripts are available in `/experiments/models/recommendation` package:
    - bert_recommender.py
    - random_recommender.py    
    - content_based_recommender.py
	- deepcbrs_recommender.py
     
## Configuration files

The configuration files are in JSON format and are composed by specific fields.
They are used in order to modify model parameters and to specify the supplementary files used to train the models or to evaluate the models. 

For instance, the configuration file for the `bert_classifier.py` file is composed by the following fields:

```json
{
	"bert_model" : "bert-base-uncased",
	"task_name" : "cbrs",
	"do_lower_case" : true,
	"train_batch_size" : 5,
	"test_batch_size" : 1,
	"gradient_accumulation_steps" : 1,
	"num_train_epochs" : 7,
	"learning_rate" : 0.000001,
	"warmup_proportion" : 0.1,
	"max_seq_length" : 512,
	"local_rank" : -1,
	"no_cuda" : false,
	"path" : "~/Semantic_aware_models/"
}
```

## License

Open source license: If you are creating an open source application under a license compatible with the GNU GPL license v3 you may use SAME under its terms and conditions.

## Contributors

Omitted temporarily due to double-blind peer review. 

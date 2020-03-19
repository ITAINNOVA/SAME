# SAME: Semantic Aware ModeEls

SAME: Semantic Aware ModeEls is a framework with different traditional recommendation models and
a semantic-aware content-based recommendation model that exploits textual features of items 
obtained from the Linked Open Data and deep learning transformers like BERT 
(Bidirectional Encoder Representations from Transformers).

## Description

In this project, we have included the following traditional recommendation models:

- A random recommendation model that predicts a random rating based on the distribution of the training set, which is assumed to be
normal. This model was provided by Surprise library.

- A traditional content-based recommendation model based on Vector Space Model (VSM) to represent the textual information of
items, by using TF-IDF weights. This model was implemented by using Sklearn library.

- A semantic-aware content-based recommendation model, based on BERT classifier able to train and test any text information with its related labels.

## Requirements

The libraries used in this project with its respective versions can be seen in `requirements.txt`.

## Usage

1. To use the BERT recommendation model, prepare the task and the input correctly. Some examples are available in `data_processors.py`.
2. Create JSON configuration file for each recommender.
3. Executable scripts are provided under the `/experiments` directory. Specifically, recommendation scripts are available in `/experiments/models/recommendation` package:
    - bert_recommender.py
    - random_recommender.py    
    - content_based_recommender.py
     
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

This product includes software developed at [ITAINNOVA](http://www.itainnova.es).

Open source license: If you are creating an open source application under a license compatible with the GNU GPL license v3 you may use SAME under its terms and conditions.

## Contributors

- María del Carmen Rodríguez Hernández - [mcrodriguez@itainnova.es](mailto:mcrodriguez@itainnova.es)
- Sergio Sabroso Lasa - [sabrosomr@gmail.com](mailto:sabrosomr@gmail.com)
- Rosa María Montañés Salas - [rmontanes@itainnova.es](mailto:rmontanes@itainnova.es)
- Rafael del Hoyo Alonso - [rdelhoyo@itainnova.es](mailto:rdelhoyo@itainnova.es)
- Sergio Ilarri - [silarri@unizar.es](mailto:silarri@unizar.es)
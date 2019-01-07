# picking-apart-story-salads

Su Wang, Eric Holgate, Greg Durrett and Katrin Erk. [Picking Apart Story Salads](http://suwangcompling.com/wp-content/uploads/2018/09/emnlp-2018-official.pdf). EMNLP 2018.

## Data

* **Wiki**: [google drive folder](https://drive.google.com/drive/folders/1-K2rRZu0nWIl37JPv4yc25NWOSps1diI?usp=sharing)

* **NYT**: copyright protected. File extractor will be uploaded soon.

## Use Pretrained Model

Download the code and place them in a single folder, then also download the (large) indexer directory in this [goole drive folder](https://drive.google.com/drive/folders/1n2yUvb0L-aVOJxzYEzI0q4aLtk7Mi__x?usp=sharing), placed in the sample folder (two directories: indexer and temp). Run `sh run_pretrained.sh` to see results on 20 samples from the Gigaword (NYT). The clustering results will appear in the `out.txt` file (sample output is prepared in the current file). Due to publishing restrictions, we will upload a document reader for the Gigaword dataset soon.

## Train Your Own Model

* **Data prep**

Prepare three objects: document a, document b, and the document mixture (details in the paper), which are lists of sentences, with each sentence being a list of words. Then use the `Indexer` object prepared (in the indexer directory, named `indexer_word2emb.p`, loaded with `indexer, word2embedding = dill.load(open(path, 'rb'))` command) to convert the words to indices. Finally pickle the index lists with `dill.dump((document_a, document_b, document_mixture), open(path, 'wb'))` and place under a folder. To run, redirect the `.sh` file by replacing the default `--data_dir` option there with the path to this folder.

* **Train pairwise classifier for clustering**

Make a shell script like this (customize directories to your own):
```
python3 pairwise_classifier.py \
  --batch_size=32 \
  --vocab_size=20000 \
  --emb_size=300 \
  --n_layer=2 \ # number of layers for stacked LSTM.
  --hid_size=100 \
  --keep_prob=0.7 \ # dropout rate.
  --learning_rate=1e-4 \
  --n_epoch=10 \
  --train_size=500000 \ # size of training data (573,681 in total).
  --verbose=1000 \ # print out every `verbose` steps.
  --save_freq=10000 \ # save model every `save_freq` steps.
  --data_dir=data/ \ # directory to data (list of lists of ints, sublist=sentence).
  --info_path=info/info.p \ # pickle file that has an Indexer and a word2emb dictionary.
  --init_with_glove=1 \ # true -> initialize embeddings with glove.
  --save_dir=saved_models/ \ # directory to save trained model.
  --save_name=my_model.ckpt \ # name of model save file.
  --restore_dir=to_load_models/ \ # directory from which model is loaded.
  --restore_name=my_model.ckpt \ # name of model to be loaded.
  --load_from_saved=1 \ # load saved model to continue training.
  --track_dir=tracks/ \ # directory to save loss/accuracy tracks.
  --new_track=1 \ # true -> keep track for a new model.
  --session_id=05182018 \ # name of a particular session.
  --mutual_attention=1 \ # apply mutual attention.
  --context=1 \ # apply context reader.
  --context_length=500 # read the first `context_length` words in the doc mixture as context.
```
Then run the clustering algorithm (i.e. K-Medoids) using a similar shell script like the one we use for running pretrained model (i.e. `run_pretrained.sh`).

# picking-apart-story-salads

Su Wang, Eric Holgate, Greg Durrett and Katrin Erk. [Picking Apart Story Salads](http://suwangcompling.com/wp-content/uploads/2018/09/emnlp-2018-official.pdf). EMNLP 2018.

**NB: the repo is under development. Convenience facilities will be added in the coming weeks!**

### USE
Download the code and place them in a single folder, then also download the (large) indexer directory in this [goole drive folder](https://drive.google.com/drive/folders/1n2yUvb0L-aVOJxzYEzI0q4aLtk7Mi__x?usp=sharing), placed in the sample folder (two directories: indexer and temp). Run `sh run_pretrained.sh` to see results on 20 samples from the Gigaword (NYT). The clustering results will appear in the `out.txt` file (sample output is prepared in the current file). Due to publishing restrictions, we will upload a document reader for the Gigaword dataset soon.

### APPLICATION TO NEW DATA
Prepare three objects: document a, document b, and the document mixture (details in the paper), which are lists of sentences, with each sentence being a list of words. Then use the `Indexer` object prepared (in the indexer directory, named `indexer_word2emb.p`, loaded with `indexer, word2embedding = dill.load(open(path, 'rb'))` command) to convert the words to indices. Finally pickle the index lists with `dill.dump((document_a, document_b, document_mixture), open(path, 'wb'))` and place under a folder. To run, redirect the `.sh` file by replacing the default `--data_dir` option there with the path to this folder.

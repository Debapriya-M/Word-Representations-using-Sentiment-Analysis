Worked upon by: Debapriya Mukherjee, Abhijat Shrivastava and Aishwarya Vijayakumar.

Repository for baseline code: https://github.com/PrashantRanjan09/Improved-Word-Embeddings

our changes ----->
On top of the existing baseline code, we have appended the biLSTM implementation of the model on a IMDB Dataset.

Files changed ----->
1. main.py (Function names: preprocess(), lemmatize(), clean_review())
2. word_embeddings.py (we changed this file initially for a sub-part but then reverted it for the last task)
3. pos_function.py (build_model())


Command used to run the model ---->

python main.py


Environment: Ubuntu(16.04.2), CUDA(9.0), python(3.5.4), pytorch(0.1.12)


Requirements:
gensim 3.8.0
fasttext-win 0.8.3
keras 2.3.1

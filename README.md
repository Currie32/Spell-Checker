# Spell-Checker

The objective of this project is to build a model that can take a sentence with spelling mistakes as input, and output the same sentence, but with the mistakes corrected. The data that we will use for this project will be twenty popular books from [Project Gutenberg](http://www.gutenberg.org/ebooks/search/?sort_order=downloads). Our model is designed using grid search to find the optimal architecture, and hyperparameter values. The best results, as measured by sequence loss with 15% of our data, were created using a two-layered network with a bi-direction RNN in the encoding layer and Bahdanau Attention in the decoding layer. [FloydHub's](https://www.floydhub.com/) GPU service was used to train the model.

All of the books that I used for training can be found in books.zip.

To view my work most easily, see the .ipynb file.

I wrote an [article](https://medium.com/@Currie32/creating-a-spell-checker-with-tensorflow-d35b23939f60) that explains how to create the input data (the sentences with spelling mistakes) for this model.

# Project

Here is the code for the first Natural Language Processing (NLP) assignment


## Subject

The classification of texts.

We had to assign one of two classes to an input text (in English only): medical/non-medical. 

We were be permitted to implement solutions based on Naive Bayes methods, both on Bag Of Words without any pre-processing or pre-processed by the SnowBall stop word list, stemming methods based on Porter's algorithm and finally lemmatization based on the WordNet Lemmatizer.
We were also be permitted to implement a solution based on Logistic Regression approach, again with a feature extraction based on pre-processing with a Naive classifier, or extracted directly from Wikipedia based on the annotated keywords. Also in this case, texts could be pre-processed or not.


## Method used

The implementation was performed by using NLTK (Python library) and is based on the Naive Bayes method.

Main steps in the code:
    - import all the necessary library (sklearn, nltk...)
    - import data from internet (thanks to fetch_20newsgroups from sklearn datasets )
    - clean data (by removing stopwords, lemmatizing and deleting punctuation)
    - convert text documents to a matrix of token counts (using CountVectorizer() from sklearn)
    - inverse document frequency (downscales words that appear a lot across documents by giving a weight to a word indicating its importance for classification)
    - do the Naive Bayes classifier (train the Multinomial Naive Bayes classifier using the TF-IDF transformed training data and the corresponding category then make predictions on the test data)


## Evaluation

The evaluation is carried out on all the categories that were used to train and test our data, even if we are only interested in the medicine category.

We then display a detailed classification report which includes precision, recall, F1-score, and support for each class in the classification task.

We also display the confusion matrix which is a table showing the number of true positives, true negatives, false positives, and false negatives.


# To test a new document

A new document can be tested to determine whether it is a medical text or not.

If it's a small sentence, it can be inserted directly into the new_documents variable. If the text is a little larger, it's better to create a new variable, put the text in front of it, and then add the text to new_documents. 

In the code we can find a few examples of how this can be done.

Next, the code predicts the category of the text and displays whether the text is medical or not. To find out which document it is, all you have to do is look at the position of the document added in the new_doc table, starting from 1 and not 0.

So now you know whether the document is a medical text or not !


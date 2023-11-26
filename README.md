# Project

Here is the code with commentary for the first Natural Language Processing (NLP) assignment


## Subject

The classification of texts.

We had to assign one of two classes to an input text (in English only): medical/non-medical. 

We were be permitted to implement solutions based on Naive Bayes methods, both on Bag Of Words without any pre-processing or pre-processed by the SnowBall stop word list, stemming methods based on Porter's algorithm and finally lemmatization based on the WordNet Lemmatizer.
We were also be permitted to implement a solution based on Logistic Regression approach, again with a feature extraction based on pre-processing with a Naive classifier, or extracted directly from Wikipedia based on the annotated keywords. Also in this case, texts could be pre-processed or not.


##  Operational Structure:

* Data Import:
    * Textual data is imported from the '20 newsgroups' dataset with categories such as 'rec.sport.baseball', 'soc.religion.christian', 'sci.med'etc.
    * These categories are used as labels to train and test the model.

* Data Cleaning:
    * A data cleaning process is performed, including the removal of stopwords, lemmatization of words, and elimination of punctuation.

* Text Vectorization (CountVectorizer):
    * Text documents are converted into a matrix of word frequency using CountVectorizer, representing the frequency of each word in the documents.

* TF-IDF Transformation:
    * A TF-IDF transformation is applied to the word frequency matrix to give more weight to words specific to each document and reduce the importance of common words.

* Model Training (Naive Bayes):
    * A multinomial Naive Bayes classification model is trained on the TF-IDF-transformed training data.

* Model Evaluation:
    * The model is evaluated on test data, and metrics such as accuracy, classification report, and confusion matrix are calculated to assess the model's performance.

* Application on New Documents:
    * The model is used to predict the categories of new documents, demonstrating its ability to generalize to unseen data.


## Pipeline:

* Data Import:
    * Using the fetch_20newsgroups function from scikit-learn to import data.

* Data Cleaning:
    * Using the clean_data_function function to clean data by removing stopwords, lemmatizing words, and eliminating punctuation.

* Text Vectorization and TF-IDF Transformation:
    * Using CountVectorizer for text vectorization.
    * Using TfidfTransformer for TF-IDF transformation.

* Model Training:
    * Using the multinomial Naive Bayes model with MultinomialNB() and training with TF-IDF-transformed training data.

* Model Evaluation:
    * Calculating accuracy, classification report, and confusion matrix to evaluate the model's performance.

* Prediction on New Documents:
    * Applying the model to new documents to predict their categories.


## Technologies Used:

Programming Language: Python
Libraries: scikit-learn, NLTK
Machine Learning Model: Multinomial Naive Bayes
Text Transformation Tools: CountVectorizer, TfidfTransformer


## Evaluation

The evaluation is carried out on all the categories that were used to train and test our data, even if we are only interested in the medicine category.

We then display a detailed classification report which includes precision, recall, F1-score, and support for each class in the classification task.

We also display the confusion matrix which is a table showing the number of true positives, true negatives, false positives, and false negatives.



## To test a new document

A new document can be tested to determine whether it is a medical text or not.

If it's a small sentence, it can be inserted directly into the new_documents variable. If the text is a little larger, it's better to create a new variable, put the text in front of it, and then add the text to new_documents. 

In the code we can find a few examples of how this can be done.

Next, the code predicts the category of the text and displays whether the text is medical or not. To find out which document it is, all you have to do is look at the position of the document added in the new_doc table, starting from 1 and not 0.

So now you know whether the document is a medical text or not !


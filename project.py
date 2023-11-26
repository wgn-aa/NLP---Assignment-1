from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


#function that cleans up data by removing stopwords, lemmatizing and deleting punctuation
def clean_data_function(data):
    stoplist = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    clean_word_list = []
    for item in data:
        remove_puntuation = "".join([char for char in item if char not in string.punctuation])
        lematize_words = " ".join([lemmatizer.lemmatize(word) for word in remove_puntuation.split()])
        remove_stopword = " ".join([word for word in lematize_words.split() if word not in stoplist])
        clean_word_list.append(remove_stopword)
    return clean_word_list



'''''''''''''''''''''''''''
IMPORT DATA FROM INTERNET
'''''''''''''''''''''''''''

categories = ['rec.sport.baseball','soc.religion.christian','comp.graphics','talk.politics.guns','rec.motorcycles','comp.os.ms-windows.misc','sci.med']
#variable which will take data from the train folder in the dataset fetch_20newsgroups
train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
#variable which will take data from the test folder
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)

print(train_data.target_names)
print("\n")

'''''''''''''''''''''
CLEAN DATA
'''''''''''''''''''''
   
clean_train_data = clean_data_function(train_data.data)
clean_test_data = clean_data_function(test_data.data)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
WORD COUNT - CONVERT TEXT DOCUMENTS TO A MATRIX OF TOKEN COUNTS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

count_vect = CountVectorizer()

#to leave the unique numbers to each word
count_vect.fit(clean_train_data)
#print("Vocabulary: ", count_vect.vocabulary_,"\n\n")   #if we want display which number is assigned to which word

#to count the number of occurrence of each word
train_data_tf = count_vect.transform(clean_train_data)

#to display more informations:
print("shape of the training data: ", train_data_tf.shape) #result (what it displays) --> (number of sample (row), number of words (column))
#print(X_train_tf) #result (what it displays) --> (n° of the row, unique numbers of the word) occurrence of the word
#print("Array: ",X_train_tf.toarray()) #to display the matrix of the word count 


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
INVERSE DOCUMENT FREQUENCY (idf) - GIVE A WEIGHT TO A WORD INDICATING ITS IMPORTANCE IN A ROW FOR CLASSIFICATION
                                 - DOWNSCALES WORDS THAT APPEAR A LOT ACROSS DOCUMENTS  
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

tfidf_transformer = TfidfTransformer()

#use fit method on transform train data
tfidf_transformer.fit(train_data_tf)

#then the transform method - word with the highest tfidf value will be the most significant and vice versa
train_data_tfidf = tfidf_transformer.transform(train_data_tf)

print("shape of the idf training data: ",train_data_tfidf.shape) #same shame as previous (no data loss)
#print(X_train_tfidf) #to display word (position) with the tfidf value


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
NAIVE BAYES CLASSIFER
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Train the Multinomial Naive Bayes classifier using the TF-IDF transformed training data and the corresponding category to make predictions later
clf = MultinomialNB().fit(train_data_tfidf, train_data.target)

#Transform the test data using the same CountVectorizer that was fit on the training data to represent test data  using the same features as the training data.
test_data_tf = count_vect.transform(clean_test_data)

# Transform the test data using the same TF-IDF transformer trained on the training data.
test_data_tfidf = tfidf_transformer.transform(test_data_tf)

#Use the trained classifier to make predictions on the TF-IDF transformed test data
predicted = clf.predict(test_data_tfidf)


'''''''''''''''''''''''''''
DISPLAYING EVALUATION - RESULTS
'''''''''''''''''''''''''''
print("\n")

# Print a detailed classification report. This report includes precision, recall, F1-score, and support for each class in the classification task
print(metrics.classification_report(test_data.target, predicted, target_names = test_data.target_names))

## Print the confusion matrix which is a table showing the number of true positives, true negatives, false positives, and false negatives.
print(metrics.confusion_matrix(test_data.target, predicted))

print("\n")


'''''''''''''''''''''''''''''''''''
TESTING NEW DOCUMENTS - if you want to know whether documents are medical text or not (insert them in the new_documents if it's all text)
'''''''''''''''''''''''''''''''''''

test_medicine_text ='''Medical availability and clinical practice vary across the world due to regional differences in culture and technology. Modern scientific medicine is highly developed in the Western world, while in developing countries such as parts of Africa or Asia, the population may rely more heavily on traditional medicine with limited evidence and efficacy and no required formal training for practitioners.[8]
In the developed world, evidence-based medicine is not universally used in clinical practice; for example, a 2007 survey of literature reviews found that about 49% of the interventions lacked sufficient evidence to support either benefit or harm.[9]
In modern clinical practice, physicians and physician assistants personally assess patients to diagnose, prognose, treat, and prevent disease using clinical judgment. The doctor-patient relationship typically begins with an interaction with an examination of the patient's medical history and medical record, followed by a medical interview[10] and a physical examination. Basic diagnostic medical devices (e.g., stethoscope, tongue depressor) are typically used. After examining for signs and interviewing for symptoms, the doctor may order medical tests (e.g., blood tests), take a biopsy, or prescribe pharmaceutical drugs or other therapies. Differential diagnosis methods help to rule out conditions based on the information provided. During the encounter, properly informing the patient of all relevant facts is an important part of the relationship and the development of trust. The medical encounter is then documented in the medical record, which is a legal document in many jurisdictions.[11] Follow-ups may be shorter but follow the same general procedure, and specialists follow a similar process. The diagnosis and treatment may take only a few minutes or a few weeks, depending on the complexity of the issue.
'''
test_non_medicine_text2 ='''A city is a human settlement of a notable size.[1][2][a] It can be defined as a permanent and densely settled place with administratively defined boundaries whose members work primarily on non-agricultural tasks.[3] Cities generally have extensive systems for housing, transportation, sanitation, utilities, land use, production of goods, and communication.[4][5] Their density facilitates interaction between people, government organizations, and businesses, sometimes benefiting different parties in the process, such as improving the efficiency of goods and service distribution.
Historically, city dwellers have been a small proportion of humanity overall, but following two centuries of unprecedented and rapid urbanization, more than half of the world population now lives in cities, which has had profound consequences for global sustainability.[6][7][8][9][10] Present-day cities usually form the core of larger metropolitan areas and urban areas—creating numerous commuters traveling toward city centres for employment, entertainment, and education. However, in a world of intensifying globalization, all cities are to varying degrees also connected globally beyond these regions. This increased influence means that cities also have significant influences on global issues, such as sustainable development, climate change, and global health. Because of these major influences on global issues, the international community has prioritized investment in sustainable cities through Sustainable Development Goal 11. Due to the efficiency of transportation and the smaller land consumption, dense cities hold the potential to have a smaller ecological footprint per inhabitant than more sparsely populated areas.[11][12] Therefore, compact cities are often referred to as a crucial element in fighting climate change.[13][14][15] However, this concentration can also have significant negative consequences, such as forming urban heat islands, concentrating pollution, and stressing water supplies and other resources.
Other important traits of cities besides population include the capital status and relative continued occupation of the city. For example, country capitals such as Athens, Beijing, Jakarta, Kuala Lumpur, London, Manila, Mexico City, Moscow, Nairobi, New Delhi, Paris, Rome, Seoul, Singapore, Tokyo, and Washington, D.C. reflect the identity and apex of their respective nations.[16] Some historic capitals, such as Kyoto, Yogyakarta, and Xi'an, maintain their reflection of cultural identity even without modern capital status. Religious holy sites offer another example of capital status within a religion; examples include Jerusalem, Mecca, Varanasi, Ayodhya, Haridwar, and Prayagraj. '''

new_documents = ['Cats love me', "Doctor love saving lives. That's why the doctor studied medicine !", test_medicine_text, test_non_medicine_text2]

clean_new_test_data = clean_data_function(new_documents)

new_data_counts = count_vect.transform(clean_new_test_data)
new_data_tfidf = tfidf_transformer.transform(new_data_counts)

prediction_new_doc = clf.predict(new_data_tfidf)


#The code iterates over the predictions which are given by numbers between 0 and 6, the category of medicine is number 4.
#Then prints the predicted category for each document 
#If the predicted category is 4 (assuming 4 corresponds to "Medical"), it prints "Medical text"; otherwise, it prints "Non-medical text".
#The loop index i is used to label each document.
i=0
for x in prediction_new_doc:
    i+=1
    if x==4:
        print(f'Doc {i} --> Medical text')
    else:
        print(f'Doc {i} --> Non-medical text')

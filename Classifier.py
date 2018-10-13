import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.externals import joblib
#import flask

def clean_df (df, col):
    df[col].replace({"'": ''}, regex = True, inplace = True)
    df[col].replace({r'[^\w\s]': ' '}, regex = True, inplace = True)
    df[col].replace({r'\s+': ' '}, regex = True, inplace = True)
    df[col] = df[col].str.lower()
    df[col] = df[col].str.strip()    
    return df

def display_cm(ytest , ypred, name_str):
    print name_str
    print (confusion_matrix(ytest , ypred))
    print ""
    print (classification_report(ytest , ypred))

df = pd.read_csv('Alexa_Review_balanced.csv')

df_train = pd.DataFrame()
df_test = pd.DataFrame()

df_train['reviews'], df_test['reviews'], df_train['rating'], df_test['rating'], = train_test_split(df.verified_reviews.values, 
                                                    df.rating.values, 
                                                    test_size=0.2, random_state=42, 
                                        stratify = df.rating.values)
                        #encode labels into numerical
df_train.drop_duplicates(subset = ['reviews'], inplace = True)
df_test.drop_duplicates(subset = ['reviews'], inplace = True)


df_train['rating'].replace({'Love': 0,'Okay':1 , 'Hate': 2}, regex = True, inplace = True)                          
df_train = clean_df(df_train, 'reviews') #step 2
tf_idf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_train = tf_idf.fit_transform(df_train['reviews'])   #Step 3

df_test['rating'].replace({'Love': 0,'Okay':1 , 'Hate': 2}, regex = True, inplace = True) #Step 1
df_test = clean_df(df_test, 'reviews') #Step 2
tfidf_test = tf_idf.transform(df_test['reviews']) #Step 3

multiNB = MultinomialNB().fit(tfidf_train, df_train.rating)
y_pred = multiNB.predict(tfidf_test)
display_cm(df_test['rating'] , y_pred, "Confusion Matrix: Multinomial Naive Bayes")


#saving preprocessor, and model instances
joblib.dump(multiNB, 'multiNB_model.pkl')
joblib.dump(tf_idf, 'tf_idf_preprocessor.pkl')

ANN = Sequential()

ANN.add(Dense(512, input_shape=tf_idf.idf_.shape))
ANN.add(Activation('relu'))
ANN.add(Dropout(0.5))

ANN.add(Dense(3))
ANN.add(Activation('softmax'))

ANN.summary()
ANN.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print "x_train:",tfidf_train.shape 
print "y_train:",df_train['rating'].shape

print "x_test:",tfidf_test.shape
print "y_test:",df_test['rating'].shape

batch_size = 15
epochs = 3

training_ANN = ANN.fit(tfidf_train, df_train['rating'], epochs=epochs, verbose=1, validation_split=0.1, batch_size = batch_size)
score = ANN.evaluate(tfidf_test, df_test['rating'], verbose=1, batch_size = batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

yp_ann = ANN.predict_classes(tfidf_test)
display_cm(df_test['rating'] , yp_ann, '\nConfusion Matrix: Neural Network')

ANN.save("ANN_model.h5")


df_test ['prediction'] = yp_ann
df_test['iscorrect'] = df_test['rating'] == df_test['prediction']

df_test['prediction'].replace({0: 'Love',1:'Okay' , 2:'Hate'}, regex = True, inplace = True) 
df_test['rating'].replace({0: 'Love',1:'Okay' , 2:'Hate'}, regex = True, inplace = True)                         

df_test.to_csv( "test_results.csv", index=False)

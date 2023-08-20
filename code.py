import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
raw_mail_data = pd.read_csv('/content/mail_data.csv')
print(raw_mail_data)

# Replace the Null values with a null string

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
mail_data.head()
# checking No of Rows and colums

mail_data.shape
# ham-->1 spam-->0 label encoding
mail_data.loc[mail_data['Category']=='spam','Category',]=0 # convert spam to 0

mail_data.loc[mail_data['Category']=='ham','Category',]=1 # convert spam to 1

# separating the data as texts and label

x=mail_data['Message']
y=mail_data['Category']
print(x) 
print(y)
# spliting data into training and testing
 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
print(x.shape)
print(x_train.shape)
print(x_test.shape)

# transform the text data to feature vectors that can be used as input to the Logistic regression
#Feature Extraction Transform Text data to Feature vectors used as input to the Logistic Regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english')

X_train_features = feature_extraction.fit_transform(x_train)
X_test_features = feature_extraction.transform(x_test)

# convert Y_train and Y_test values as integers

y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(x_train)

# Training with the model

model = LogisticRegression()
model.fit(X_train_features,y_train)
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_train_data = accuracy_score(y_train,prediction_on_training_data)
print('Accuracy on Train data: ',accuracy_on_train_data)
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(y_test,prediction_on_test_data)
print('Accuracy on Test data: ',accuracy_on_test_data)
input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')

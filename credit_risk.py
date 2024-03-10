import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing 
from sklearn.metrics import classification_report, confusion_matrix

credit_data = pd.read_csv("german_credit_data.csv")

# Exploring the credit dataset
credit_data.describe(include= 'all')
credit_data.info()


# Dropping the extra index column and replacing nan with 'unknown'. 
# Also replacing numbers with job type in 'Job' column to understand the correlation better.
credit_data = credit_data.drop('Unnamed: 0', axis=1)
credit_data[['Saving accounts', 'Checking account']] = credit_data[['Saving accounts', 'Checking account']].fillna('unknown')

credit_data["Job"]= credit_data["Job"].astype(str)
credit_data['Job'].replace('0', 'unskilled and non-resident', inplace=True)
credit_data['Job'].replace('1', 'unskilled and resident', inplace=True)
credit_data['Job'].replace('2', 'skilled', inplace=True)
credit_data['Job'].replace('3', 'highly skilled', inplace=True)

# Creating histograms to look at the distrubutions of 'Risk' by various columns

#risk and saving accounts
count_table = credit_data[['Saving accounts', 'Risk']].value_counts().unstack()

fig, ax = plt.subplots(figsize = (8,6))
ax.bar(count_table.index, count_table["bad"], label = "bad", width = 0.8) 
ax.bar(count_table.index, count_table["good"], bottom = count_table.bad, label = "good", width = 0.8) 
ax.legend(labels = ["bad", "good"],fontsize = 10,title = "Risk",title_fontsize = 14,bbox_to_anchor = [0.55, 0.7])
ax.set_xlabel("Saving Accounts", size = 12)

for c in ax.containers:
    labels = [str(round(v.get_height(), 2)) if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c,
                 label_type='center',
                 labels = labels,
                 size = 9)


#risk and checking account
count_table = credit_data[['Checking account', 'Risk']].value_counts().unstack()

fig, ax = plt.subplots(figsize = (8,6))
ax.bar(count_table.index, count_table["bad"], label = "bad", width = 0.8) 
ax.bar(count_table.index, count_table["good"], bottom = count_table.bad, label = "good", width = 0.8) 
ax.legend(labels = ["bad", "good"],fontsize = 10,title = "Risk",title_fontsize = 14,bbox_to_anchor = [0.55, 0.7])
ax.set_xlabel("Checking Account", size = 12)

for c in ax.containers:
    labels = [str(round(v.get_height(), 2)) if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c,
                 label_type='center',
                 labels = labels,
                 size = 9)
    
#job and Risk 
count_table = credit_data[['Job', 'Risk']].value_counts().unstack()

fig, ax = plt.subplots(figsize = (12,6))
ax.bar(count_table.index, count_table["bad"], label = "bad", width = 0.8) 
ax.bar(count_table.index, count_table["good"], bottom = count_table.bad, label = "good", width = 0.8) 
ax.legend(labels = ["bad", "good"],fontsize = 10,title = "Risk",title_fontsize = 14,bbox_to_anchor = [0.55, 0.7])
ax.set_xlabel("Job Type", size = 12)

for c in ax.containers:
    labels = [str(round(v.get_height(), 2)) if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c,
                 label_type='center',
                 labels = labels,
                 size = 9)   
plt.show()

#credit amount and Risk
count_table = credit_data.pivot(columns='Risk', values='Credit amount').reset_index()
# plt.hist(count_table['bad'], alpha=0.5, label='bad')
# plt.hist(count_table['good'],alpha=0.5, label='good')
plt.hist((count_table['bad'],count_table['good']), stacked = True, label= ('bad','good'))
plt.legend()
plt.xlabel("Credit Amount")
plt.show()


#label endcoding: converting the categorical variables into numerical
label_encoder = preprocessing.LabelEncoder() 
credit_data['Sex']= label_encoder.fit_transform(credit_data['Sex']) 
credit_data['Job']= label_encoder.fit_transform(credit_data['Job']) 
credit_data['Housing']= label_encoder.fit_transform(credit_data['Housing']) 
credit_data['Saving accounts']= label_encoder.fit_transform(credit_data['Saving accounts']) 
credit_data['Checking account']= label_encoder.fit_transform(credit_data['Checking account']) 
credit_data['Purpose']= label_encoder.fit_transform(credit_data['Purpose']) 

#Shuffling the dataset and splitting it into test and train data 
X = credit_data.iloc[:, :-1].values
y = credit_data.iloc[:, -1].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)


# Standardizing the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Using the KNN classifier to fit data and find the optimal value of k.
accuracy_scores = []
knn = [i for i in range (1,51)]
for n in knn:
    classifier = KNeighborsClassifier(n_neighbors=n)
    classifier.fit(X_train, y_train) 
    # Predict y data with classifier: 
    y_predict = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    accuracy_scores.append(accuracy)

# plotting the k values along with the accuracy scores
plt.plot(knn, accuracy_scores, 'ok-')
plt.xlabel('Knn Values')
plt.ylabel('Accuracy Scores')
plt.show()

# Finding the k value with max accuracy score
op_index = np.argmax(accuracy_scores)
op_k = knn[op_index]


# Using the KNN classifier to fit data with the optimal k value.
classifier = KNeighborsClassifier(n_neighbors=op_k)
classifier.fit(X_train, y_train) 
y_predict = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy Score for the best k:", accuracy)

#confusion matrix and classification matrix 
#precision: based on predicted values
#recall: based on actual values
confusion_matrix = confusion_matrix(y_test, y_predict)
classification_report = classification_report(y_test, y_predict)
print("Confusion Matrix:", confusion_matrix)
print("Classification Report:", classification_report)





























































































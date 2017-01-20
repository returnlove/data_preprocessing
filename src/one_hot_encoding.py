import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale

data_path = "../data/"
Xtrain_path = data_path + "X_train.csv"
ytrain_path = data_path + "Y_train.csv"
Xtest_path = data_path + "X_test.csv"
ytest_path = data_path + "Y_test.csv"

X_train = pd.read_csv(Xtrain_path)
y_train = pd.read_csv(ytrain_path)

X_test = pd.read_csv(Xtest_path)
y_test = pd.read_csv(ytest_path)


# print(X_train.head())
# print(y_train.head())

# clf = LogisticRegression(penalty='l2',C=.01)
# clf.fit(X_train, y_train)
# print(accuracy_score(y_test, clf.predict(X_test)))


le = LabelEncoder()
# print(X_test.columns.values)
# print('before')
# print(X_train.shape)
# print(X_test.shape)
# print(X_train.head())

for col in X_test.columns.values:
	if(X_train[col].dtypes == 'object'):
		# print(col)
		# print(X_test[col].dtypes)

		# to get all the levels
		# print(len(X_train[col]))
		# print(len(X_test[col]))
		data = X_train[col].append(X_test[col])
		# print(len(data))
		# print(data.values)
		le.fit(data.values)
		X_train[col] = le.transform(X_train[col])
		X_test[col] = le.transform(X_test[col])
		# print(X_train[col])

# print('after')

# print(X_train.shape)
# print(X_test.shape)
# print(X_train.head())

# clf = LogisticRegression()
# clf.fit(X_train, y_train)
# print(accuracy_score(y_test, clf.predict(X_test)))

# one hot encoding

enc = OneHotEncoder(sparse = False)
X_train1 = X_train
X_test1 = X_test

# lets find out the features with categorical type
# categorical_columns = []
# for col in X_train1.columns.values:
# 	# print(col)
# 	if X_train1[col].dtypes == 'object':
# 		# print('true')
# 		categorical_columns.append(col)

# print(categorical_columns)
# print(X_train1.Credit_History.dtypes)
# print(X_train1.Credit_History.head())

# categorical_columns.remove("Loan_ID")
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area']
print(categorical_columns)

	# if(X_train[col].dtypes == 'object'):
# print('X_train1')
# print(X_train1.head())
# print(X_train1.shape)
# print(X_train1.size, '====', 384*12)
# print(len(X_train1.columns))
for col in categorical_columns:
	# print(col)
	data=X_train[[col]].append(X_test[[col]])
	enc.fit(data)
	# print('error')
	temp = enc.transform(X_train[[col]])
	# print('transform ok')
	# print(data[col].value_counts())
	# print(data[col].value_counts().index)

	# temp = pd.DataFrame(temp, columns = [col+"_"+data[col].value_counts().index])
	temp = pd.DataFrame(temp, columns = [col+"_"+str(i) for i in data[col].value_counts().index])

	# setting up similar index values as train
	# print('index values of ' + col)
	# print(X_train.index.values)
	temp = temp.set_index(X_train.index.values)
	X_train1=pd.concat([X_train1,temp],axis=1)

	# fitting one hot on test data
	temp = enc.transform(X_test[[col]])
	# print('transform' + col + 'ok')
	# print(temp)

	temp = pd.DataFrame(temp, columns = [col+"_"+str(i) for i in data[col].value_counts().index])
	temp = temp.set_index(X_test.index.values)
	X_test1=pd.concat([X_test1,temp],axis=1)






# print('after X_train1: ')
# print(X_train1.head())
# print(X_train1.shape)
# print(len(X_train1.columns))
# print(X_train1.size, '====', 384*29)


	

X_train_scale = scale(X_train1)
X_test_scale = scale(X_test1)

clf = LogisticRegression(penalty='l2',C=1)
clf.fit(X_train_scale,y_train)
print(accuracy_score(y_test, clf.predict(X_test_scale)))

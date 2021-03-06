import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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
# print(X_train.shape)
# print(y_train.head())

# clf = LogisticRegression(penalty='l2',C=.01)
# clf.fit(X_train, y_train)
# print(accuracy_score(y_test, clf.predict(X_test)))


le = LabelEncoder()
# print(X_test.columns.values)
print('before')
print(X_train.shape)
print(X_test.shape)
print(X_train.head())

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

print('after')

print(X_train.shape)
print(X_test.shape)
print(X_train.head())

clf = LogisticRegression()
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))





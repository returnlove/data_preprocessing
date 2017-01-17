import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
def main():

	data_path = "../data/"
	X_train_path = data_path + "X_train.csv"
	X_train_data = pd.read_csv(X_train_path)
	y_train_path = data_path + "y_train.csv"
	y_train_data = pd.read_csv(y_train_path)

	X_test_path = data_path + "X_test.csv"
	X_test_data = pd.read_csv(X_test_path)
	y_test_path = data_path + "y_test.csv"
	y_test_data = pd.read_csv(y_test_path)

	print(X_train_data.head())


	X_train_data[X_train_data.dtypes[(X_train_data.dtypes=="float64")|(X_train_data.dtypes=="int64")].index.values].hist(figsize=[11,11])
	# plt.show()



	# performing LR before normalizing the features
	clf = LogisticRegression(penalty='l2',C=0.01)
	# print(X_train_data[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']].shape)
	# print(y_train_data.shape)
	# print(y_train_data.values.ravel())
	clf.fit(X_train_data[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']],y_train_data.values.ravel())

	# check peformance on test data
	print(accuracy_score(y_test_data, clf.predict(X_test_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])))

	# check the dist of loan staus in train data set
	print(y_train_data.Target.value_counts()/y_train_data.Target.count())
	# there are 70% of the loans which are approved

	#lets assume all the loans are approved and check the predictions
	print(y_test_data.Target.value_counts()/y_test_data.Target.count())
	# 63% of the loans are approved, so we get 63% accuracy if we assume all the loans are approved which is greater than the accuracy we received earlier



	X_train_min_max = scale(X_train_data[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']])
	X_test_min_max = scale(X_test_data[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']])	


	clf = LogisticRegression(penalty='l2',C=0.01)
	clf.fit(X_train_min_max, y_train_data.values.ravel())
	print(accuracy_score(y_test_data, clf.predict(X_test_min_max)))
	# now the accuracy has increased from 61 to 75, because we normalized the features so features will not dominate
	# if their range is large compared to smaller ones

if __name__ == "__main__":
	main()
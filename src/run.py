import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

	# performing KNN before normalizing the features
	clf = KNeighborsClassifier(n_neighbors =5)
	# print(X_train_data[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']].shape)
	# print(y_train_data.shape)
	# print(y_train_data.values.ravel())
	clf.fit(X_train_data[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']],y_train_data.values.ravel())

	# check peformance on test data
	print(accuracy_score(y_test_data, clf.predict(X_test_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])))



if __name__ == "__main__":
	main()
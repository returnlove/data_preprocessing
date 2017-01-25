from sklearn.preprocessing import OneHotEncoder
import pandas as pd

enc = OneHotEncoder()  
df = pd.DataFrame([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
print(df)
enc.fit(df) 
temp = enc.transform([[0, 1, 3]]).toarray()
print(temp)
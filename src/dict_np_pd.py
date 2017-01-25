import numpy as np
import pandas as pd

my_dict = {'name':['a','b','c'], 'age':[1,2,3]}
print(my_dict)
print(pd.DataFrame(my_dict))

df = pd.DataFrame([['a',1],['b',2],['c',3]], columns = ['name','age'])
print(df)

np_arr = np.array([['a',1],['b',2],['c',3]])
print(np_arr)
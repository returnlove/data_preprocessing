import pandas as pd
import numpy as np

# to create a 3rd column from 2nd column and if nan then from 1st column


my_dict = {'name': ['a', 'b', 'c'], 'age': [np.nan, 25, np.nan], 'age1': [1, 10, 20]}
# print(my_dict)

df = pd.DataFrame(my_dict)
# print(df)

# duplicate age column
df['new_age'] = df['age']


# df[df['age'].isnull()]['new_age'] = df[df['age'].isnull()]['age1']

# select all the rows of new_age column where age is null and assign age1 by selecting the rows where age is null
df.loc[df['age'].isnull(), 'new_age'] = df[df['age'].isnull()]['age1']

print(df)


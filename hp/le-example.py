from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1) Basic usage on a 1D label array (what LabelEncoder is for)
y_train = np.array(["cat", "dog", "dog", "mouse"])
le = LabelEncoder().fit(y_train)

print(le.classes_)                # ['cat' 'dog' 'mouse']
print(le.transform(y_train))      # [0 1 1 2]
print(le.inverse_transform([2]))  # ['mouse']

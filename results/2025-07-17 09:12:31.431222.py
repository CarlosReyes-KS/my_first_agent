import pandas as pd
from sklearn.model_selection import train_test_split
import pycaret.classification

data = pd.read_csv('creditcard.csv')
X = data.drop(['Time', 'Amount'], axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = classification.create(data='X_train', target='y_train', strategy='auto_classify')

clf.fit()

predictions = clf.predict(X_test)
print('Accuracy: ', clf.evaluate().get('accuracy'))

import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import *
data = pd.read_csv('creditcard.csv')
X = data.drop(['Time', 'Amount'], axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
setup(data=X_train, target='Class', session_id=123)
model_info = compare_models()
best_model = choose_model(model_info, 'accuracy')
fit(best_model)
predicted = predict(best_model, X_test)
evaluate(y_test, predicted)

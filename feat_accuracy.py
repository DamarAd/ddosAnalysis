from normalize import *

# Create a new random forest classifier for the most important features
rfc_important = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
rfc_important.fit(X_important_train, Y_train)

from sklearn.metrics import accuracy_score

# Apply The Full Featured Classifier To The Test Data
Y_pred = rfc.predict(X_test)

# View The Accuracy Of Our Full Feature (4 Features) Model
accuracy_score(Y_test, Y_pred)

# Apply The Full Featured Classifier To The Test Data
Y_important_pred = rfc_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
accuracy_score(Y_test, Y_important_pred)
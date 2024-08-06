from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def sk_learn_gradient_boosting(X_train, y_train, X_test, y_test):
    gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)

    gb_model.fit(X_train, y_train)

    # Make predictions on the test set using  best model
    y_pred_gb = gb_model.predict(X_test)

    # Calculate the accuracy of the Gradient Boosting model with the best hyperparameters
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print("Gradient Boosting Accuracy:", accuracy_gb)
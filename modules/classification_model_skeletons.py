from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier



def sk_learn_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train sk_learns GB model implementation on input data, evaluate accuracy

    Args:
        X_train (array-like): X (feature-only) training data
        y_train (array-like): Y (label-only) training data
        X_test (array-like): X (feature-only) testing data
        y_test (array-like): Y (label-only) testing data
        ***TO-DO: add parameters for model parameters***

    Returns:
        float: model accuracy

    """
    
    gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)

    print('fitting model')
    gb_model.fit(X_train, y_train)

    print('model fitted, now making predictions on test data')
    y_pred_gb = gb_model.predict(X_test)

    accuracy_gb = accuracy_score(y_test, y_pred_gb)

    return accuracy_gb



def sk_learn_MLP(X_train, y_train, X_test, y_test, hidden_layer_size):
    """Train sk_learns MLP model implementation on input data, evaluate accuracy

    Args:
        X_train (array-like): X (feature-only) training data
        y_train (array-like): Y (label-only) training data
        X_test (array-like): X (feature-only) testing data
        y_test (array-like): Y (label-only) testing data
        hidden_layer_size: 
        ***TO-DO: add more parameters for model parameters***

    Returns:
        float: model accuracy
        
    """

    mlpModel = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), random_state=42, max_iter=800)

    print('fitting model')
    mlpModel.fit(X_train, y_train)

    print('model fitted, now making predictions on test data')
    model_y_predict = mlpModel.predict(X_test)

    accuracy_mlp = accuracy_score(y_test, model_y_predict)

    return accuracy_mlp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold


IMB = 1.0


def calculate_uniform_feature_ratio(features):
    count_uniform = 0
    total_count = len(features)
    
    for feature in features:
        if all(x == feature[0] for x in feature):
            count_uniform += 1
    if total_count > 0:
        ratio = count_uniform / total_count * 100
    else:
        ratio = 0 

    return ratio


def train_and_evaluate_model(train_features, train_labels, test_features, test_labels):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)
    print("Data for model ready!")
    
    model = LogisticRegression(max_iter=100000, solver='saga')
    param_grid = {
        'C': [0.01, 1, 10],
        'penalty': ['l1', 'l2']
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid_search = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, train_labels)

    best_model = grid_search.best_estimator_
    y_scores = best_model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(test_labels, y_scores)
    # print(f"AUC: {auc_roc}")

    return auc_roc, calculate_uniform_feature_ratio(train_features) 
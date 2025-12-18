from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, accuracy_score

SCORERS = {"f1": "f1_macro", "acc": "balanced_accuracy"}

def train_eval(pipeline, X_train, X_test, y_train, y_test, pca=False):
    cv_scores = cross_validate(
        pipeline, X_train, y_train,
        cv=5, scoring=SCORERS, n_jobs=-1
    )
    cv_f1 = cv_scores['test_f1'].mean()
    cv_acc = cv_scores['test_acc'].mean()

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    test_f1 = f1_score(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)

    return {"pipeline": pipeline,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "cv_f1": cv_f1,
            "cv_acc": cv_acc}


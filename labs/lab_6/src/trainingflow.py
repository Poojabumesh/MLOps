from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class TrainingFlow(FlowSpec):

    cv_folds = Parameter('cv_folds', default=3)

    @step
    def start(self):
        X, y = datasets.load_wine(return_X_y=True)
        print("Data loaded successfully")
        self.X = X
        self.y = y
        self.next(self.split)

    @step
    def split(self):
        # 3) Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.next(self.tune)

    @step
    def tune(self):
        # 4) Hyperparameter tuning
        rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        gs = GridSearchCV(rf, param_grid, cv=self.cv_folds)
        gs.fit(self.X_train, self.y_train)
        self.best_model = gs.best_estimator_
        self.best_score = gs.best_score_
        self.next(self.register)

    @step
    def register(self):
        # 5) Register & save best model in MLflow
        mlflow.set_experiment("metaflow-wine")
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                self.best_model,
                artifact_path='model',
                registered_model_name='WineRF'
            )
            mlflow.log_params(self.best_model.get_params())
            mlflow.log_metric("best_cv_score", self.best_score)
        self.next(self.end)

    @step
    def end(self):
        print("Training complete. Model registered in MLflow.")

if __name__ == '__main__':
    TrainingFlow()


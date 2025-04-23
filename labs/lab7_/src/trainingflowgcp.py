from metaflow import FlowSpec, step, kubernetes, resources, timeout, retry, catch
import os, mlflow, mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#@conda_base(python="3.9.16", libraries={
#    "numpy":"1.23.5","scikit-learn":"1.2.2","mlflow":">=2.21.0"
#})
class TrainingFlowGCP(FlowSpec):

    @step
    def start(self):
        # point MLflow at your Cloud Run server
        mlflow.set_tracking_uri(os.environ['MLFLOW_URL'])
        mlflow.set_experiment("metaflow-wine-gcp")
        # load data
        X,y = datasets.load_wine(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = \
          train_test_split(X,y,test_size=0.2,random_state=42)
        self.next(self.tune)

    @kubernetes(cpu=2, memory="4Gi")
    @resources(cpu=2, memory=4000)
    @timeout(seconds=1800)
    @retry(times=2)
    @catch(var='tune_error')
    @step
    def tune(self):
        # hyperparameter search
        rf = RandomForestClassifier(random_state=42)
        gs = GridSearchCV(rf, {
            'n_estimators':[50,100,200],
            'max_depth':[3,5,10,None]
        }, cv=3)
        gs.fit(self.X_train, self.y_train)
        self.best_model = gs.best_estimator_
        self.best_score = gs.best_score_
        self.next(self.register)

    @timeout(seconds=600)
    @retry(times=3)
    @step
    def register(self):
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                self.best_model,
                artifact_path='model',
                registered_model_name='WineRF_GCP'
            )
            mlflow.log_metric("best_cv_score", self.best_score)
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Training done — model registered as WineRF_GCP")

if __name__=='__main__':
    TrainingFlowGCP()


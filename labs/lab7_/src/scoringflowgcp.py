# scoringflowgcp.py

from metaflow import FlowSpec, step, resources, timeout, retry, catch
from metaflow import Parameter
import os
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ScoringFlowGCP(FlowSpec):

    model_name    = Parameter('model_name',    default='WineRF_GCP')
    model_version = Parameter('model_version', default='latest')

    @step
    def start(self):
        # ← STEP 1: point MLflow client at your server:
        mlflow.set_tracking_uri(os.environ['MLFLOW_URL'])
        # ← STEP 2: name (or create) the experiment:
        mlflow.set_experiment('metaflow-wine-gcp_scoring')

        # load test data
        X, y = datasets.load_wine(return_X_y=True)
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("✅ Test data loaded successfully")
        self.next(self.load_model)

    @timeout(seconds=600)
    @retry(times=3)
    @catch(var='model_error')
    @step
    def load_model(self):
        if hasattr(self, 'model_error'):
            print(f"⚠️ Previous loading error: {self.model_error}")
        uri = f"models:/{self.model_name}/{self.model_version}"
        print(f"🔍 Loading model from {uri}")
        self.model = mlflow.sklearn.load_model(uri)
        print("✅ Model loaded")
        self.next(self.score)

    @resources(cpu=1, memory=2000)
    @timeout(seconds=600)
    @retry(times=2)
    @step
    def score(self):
        # make predictions
        preds = self.model.predict(self.X_test)
        acc   = accuracy_score(self.y_test, preds)
        report = classification_report(self.y_test, preds)

        # ← STEP 3: actually create an MLflow run & log:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"📝 Start MLflow run: {run_id}")
            mlflow.log_metric("test_accuracy", acc)
            # write and log the text report
            with open('report.txt','w') as f:
                f.write(report)
            mlflow.log_artifact('report.txt')
        print(f"✅ Finished MLflow run {run_id}")

        self.accuracy = acc
        self.next(self.end)

    @step
    def end(self):
        print(f"🏁 Done. Test accuracy = {self.accuracy:.4f}")
        print("➡️  Go to your MLflow UI under 'metaflow-wine-gcp_scoring' to inspect the run.")

if __name__ == '__main__':
    ScoringFlowGCP()


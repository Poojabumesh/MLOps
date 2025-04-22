from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd

class ScoringFlow(FlowSpec):

    # Path to new data CSV
    input_csv = Parameter('input_csv', help="Path to new data", default="data/new_wine.csv")

    @step
    def start(self):
        # 1) Ingest new data
        self.new_df = pd.read_csv(self.input_csv)
        self.next(self.load_model)

    @step
    def load_model(self):
        # 2) Load the latest registered model
        client = mlflow.tracking.MlflowClient()
        # get latest version of your model
        versions = client.get_latest_versions("WineRF", stages=["None","Production"])
        self.model_uri = versions[-1].source  # or use .run_id + /artifacts/model
        self.model = mlflow.sklearn.load_model(self.model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        # 3) Make predictions
        X_new = self.new_df  # must match features used in training
        self.preds = self.model.predict(X_new)
        self.next(self.end)

    @step
    def end(self):
        # 4) Output or save
        self.new_df['prediction'] = self.preds
        print(self.new_df.head())
        # Optionally write to CSV:
        self.new_df.to_csv("scored_output.csv", index=False)
        print("Scoring complete, saved to scored_output.csv")

if __name__ == '__main__':
    ScoringFlow()


import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "nyc-taxi-hw-experiment"
EXPERIMENT_NAME = "nyc-taxi-hw-experiment"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
#mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        # Inside run_train, after loading data
        sample_fraction = 0.1 # Use 10% of the data
        train_sample_size = int(X_train.shape[0] * sample_fraction)

        print(f"Using a sample of {train_sample_size} for training due to potential memory issues...")
        X_train_sample = X_train[:train_sample_size]
        y_train_sample = y_train[:train_sample_size]

        print("Fitting model with sampled data...")
        rf.fit(X_train_sample, y_train_sample)
        mlflow.sklearn.log_model(rf,"model")
        # Evaluate model on the validation and test sets
        sample_fraction = 0.1 # Use 10% of the data
        val_sample_size = int(X_val.shape[0] * sample_fraction)

        print(f"Using a sample of {val_sample_size} for validation due to potential memory issues...")
        X_val_sample = X_val[:val_sample_size]
        y_val_sample = y_val[:val_sample_size]
        val_rmse = root_mean_squared_error(y_val_sample, rf.predict(X_val_sample))
        print(f"val_rmse: {val_rmse}")
        mlflow.log_metric("val_rmse", val_rmse)

        sample_fraction = 0.1 # Use 10% of the data
        test_sample_size = int(X_test.shape[0] * sample_fraction)

        print(f"Using a sample of {test_sample_size} for testing due to potential memory issues...")
        X_test_sample = X_test[:test_sample_size]
        y_test_sample = y_test[:test_sample_size]
        test_rmse = root_mean_squared_error(y_test_sample, rf.predict(X_test_sample))
        mlflow.log_metric("test_rmse", test_rmse)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)
        print(f"Registered {len(runs)} models from the top {top_n} runs.")
    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    # best_run = client.search_runs( ...  )[0]
    runs = client.search_runs(experiment_ids=experiment.experiment_id,max_results=1,
                              order_by=['metrics.test_rmse ASC'])
    print(runs)

    # Register the best model
    best_run_id = runs[0].info.run_id
    mlflow.register_model(model_uri=f"runs:/{best_run_id}/model",name='nyc-registered-hw-model')


if __name__ == '__main__':
    mlflow.autolog(disable=True)
    run_register_model()

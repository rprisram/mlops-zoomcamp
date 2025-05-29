import os
import pickle
import click
import mlflow
import traceback # Import traceback

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    print(f"Attempting to load pickle file: {filename}")
    if not os.path.exists(filename):
        print(f"ERROR: File not found at {filename}")
        raise FileNotFoundError(f"No such file or directory: {filename}")
    with open(filename, "rb") as f_in:
        content = pickle.load(f_in)
        print(f"Successfully loaded {filename}")
        return content

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    print(f"--- Starting run_train with data_path: {data_path} ---")
    
    try:
        print("Attempting to start MLflow run...")
        with mlflow.start_run():
            print("MLflow run started successfully.")
            
            train_file_path = os.path.join(data_path, "train.pkl")
            val_file_path = os.path.join(data_path, "val.pkl")

            print(f"Loading training data from: {train_file_path}")
            X_train, y_train = load_pickle(train_file_path)
            print("Training data loaded.")

            print(f"Loading validation data from: {val_file_path}")
            X_val, y_val = load_pickle(val_file_path)
            print("Validation data loaded.")

            print("Initializing RandomForestRegressor...")
            rf = RandomForestRegressor(max_depth=10, random_state=0)
            print("RandomForestRegressor initialized.")
            # Inside run_train, after loading data
            sample_fraction = 0.1 # Use 10% of the data
            train_sample_size = int(X_train.shape[0] * sample_fraction)

            print(f"Using a sample of {train_sample_size} for training due to potential memory issues...")
            X_train_sample = X_train[:train_sample_size]
            y_train_sample = y_train[:train_sample_size]

            print("Fitting model with sampled data...")
            rf.fit(X_train_sample, y_train_sample)
            print("Model fitting with sampled data completed.")
            print("Fitting model...")
            #rf.fit(X_train, y_train)
            print("Model fitting completed.")

            print("Making predictions...")
            y_pred = rf.predict(X_val)
            print("Predictions made.")

            print("Calculating RMSE...")
            rmse = root_mean_squared_error(y_val, y_pred)
            print("RMSE calculation completed.")

            # This is the print statement you were expecting
            print(f"FINAL RMSE: {rmse}")

            print("Logging metrics, model, params, and tags to MLflow...")
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(rf, "model") # Changed "model" to a more descriptive name perhaps "random-forest-model"
            mlflow.log_params(rf.get_params()) # Autolog might do this, but explicit is fine too
            mlflow.set_tag("model_type", "RandomForestRegressor")
            print("MLflow logging calls made.")
            
    except Exception as e:
        print(f"!!! AN ERROR OCCURRED IN run_train !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Traceback:")
        traceback.print_exc() # This will print the full traceback
        # Re-raise the exception if you want the script to exit with an error code
        # or if you want MLflow to automatically mark the run as FAILED if it's inside the 'with mlflow.start_run()'
        # If the error is before mlflow.start_run(), then this try-except is critical.
        # If inside, MLflow's context manager should handle run status.
        raise
    finally:
        print("--- run_train function finished or exited due to error ---")


if __name__ == '__main__':
    print("--- Script execution started ---")
    try:
        print("Setting MLflow tracking URI...")
        mlflow.set_tracking_uri("sqlite:///mlflow1.db")
        print("Setting MLflow experiment...")
        mlflow.set_experiment("nyc-taxi-hw-experiment")
        print("Enabling MLflow autolog...")
        mlflow.autolog() # Consider moving this inside run_train, after mlflow.start_run() if issues persist
        print("MLflow setup complete.")
        
        run_train()
        
    except Exception as e:
        # This outer try-except will catch errors from run_train if not caught inside,
        # or errors from the MLflow setup itself.
        print(f"!!! AN ERROR OCCURRED IN __main__ !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Traceback:")
        traceback.print_exc()
    finally:
        print("Attempting to ensure MLflow run is ended if active...")
        # This ensures end_run is called even if run_train raises an error
        if mlflow.active_run(): # Check if a run is still active
             print(f"MLflow run '{mlflow.active_run().info.run_id}' is active, attempting to end it.")
             mlflow.end_run()
             print("mlflow.end_run() called.")
        else:
            print("No active MLflow run to end.")
        print("--- Script execution finished ---")
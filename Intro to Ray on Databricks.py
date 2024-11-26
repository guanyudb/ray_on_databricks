# Databricks notebook source
# MAGIC %md
# MAGIC ## Intro to Ray on Databricks
# MAGIC Version: 1.0 11/6/2024 \
# MAGIC DBR 15.4 LTS ML \
# MAGIC guanyu.chen@databricks.com 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1: Ray Cluster Setup
# MAGIC
# MAGIC ### Objective
# MAGIC Set up a local or distributed Ray cluster, which forms the foundation for running Ray tasks. We’ll configure the cluster to handle basic parallel processing, which can be scaled for distributed computing if needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Install Ray
# MAGIC **1.Starting MLR 15.0 onwards, Ray is preinstalled on Databricks clusters.** \
# MAGIC **2.Only clusters with “no-isolation-shared” or “assigned” mode are supported.** \
# MAGIC **3.As of November 2024, Ray is not supported on Serverless compute cluster, but Serverless Ray is on road map.** 
# MAGIC
# MAGIC If Ray is not already installed, install it by running:
# MAGIC ```bash
# MAGIC pip install ray[default]
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Setup and Initialize a Ray Cluster
# MAGIC
# MAGIC Here’s how to set up a Ray cluster locally:\
# MAGIC The below experiment cluster uses Standard_D8ads_v5 with 5 nodes including 1 driver node and 4 worker nodes, each node has 8 cores and 32GB memory.

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, MAX_NUM_WORKER_NODES, shutdown_ray_cluster
import ray
setup_ray_cluster(min_worker_nodes=1, 
                  max_worker_nodes=5,
                  num_cpus_head_node=8, 
                  num_gpus_worker_node=0, 
                  num_cpus_worker_node=8, 
                  num_gpus_head_node=0)

# Initialize Ray
ray.init(ignore_reinit_error=True)

# COMMAND ----------

# Confirm the Ray setup by checking node information
print("Ray cluster resources:", ray.cluster_resources())

# COMMAND ----------

import ray._private.state as state

# Print current Ray cluster status
print("Current Ray cluster status:")
print(state.nodes())

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**: Monitoring helps ensure optimal cluster utilization. `ray._private.state.nodes()` provides information about active nodes in the cluster.
# MAGIC
# MAGIC ### References
# MAGIC - [Ray Documentation - Cluster Setup](https://docs.ray.io/en/latest/cluster/index.html)
# MAGIC - [Ray Monitoring and Debugging](https://docs.ray.io/en/latest/ray-core/debugging.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Different Ray Core Classes with Examples
# MAGIC
# MAGIC ### Objective
# MAGIC Learn how to define and utilize different classes in Ray to parallelize tasks. We will define a class and use the `@ray.remote` decorator to run its methods concurrently across multiple actors, simulating parallel data processing in a healthcare context.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ray Data Examples

# COMMAND ----------

import ray.data

# in order to use ray data, set the below in your cluster configuration
# spark.conf.set("spark.databricks.pyspark.dataFrameChunk.enabled", "true")

source_table = "samples.nyctaxi.trips"

# Read a Spark DataFrame from a Delta table in Unity Catalog
df = spark.read.table(source_table)
ray_ds = ray.data.from_spark(df)
print(ray_ds)

# write to uc table
# ray.data.write_databricks_table()

# COMMAND ----------

import os
import ray

access_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host_url = f"{dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get('browserHostName').get()}"

# Set Databricks token and host environment variables using notebook authentication
os.environ['DATABRICKS_TOKEN'] = access_token
os.environ['DATABRICKS_HOST'] = host_url

ds = ray.data.read_databricks_tables(
    warehouse_id='ddd42373c356f148',
    catalog='samples',
    schema='nyctaxi',
    query='select * from trips',
)

# ds = ray.data.read_delta_sharing_tables(
#     url=f"your-profile.json#your-share-name.your-schema-name.your-table-name",
#     limit=100000,
#     version=1,
# )

print(ds)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Parallel Processing with Tasks
# MAGIC We’ll use a simple function to simulate processing multiple chunks of healthcare data in parallel.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Define a Remote Task
# MAGIC
# MAGIC Define a function with `@ray.remote` to make it a Ray task. This function will simulate processing a chunk of healthcare data.
# MAGIC

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
import numpy as np

@ray.remote
def normalize_list(input_list, feature_range=(0, 1)):
    # Reshape the list to fit the scaler's requirements
    input_array = np.array(input_list).reshape(-1, 1)
    
    # Initialize the scaler with the specified feature range
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit and transform the data
    scaled_array = scaler.fit_transform(input_array)
    
    # Flatten the result back to a 1D list
    scaled_list = scaled_array.flatten().tolist()
    
    return scaled_list

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**: `process_chunk` is a simple function that takes a list (or chunk) of healthcare data and performs a processing operation. By adding `@ray.remote`, we convert it to a Ray task, enabling parallel execution.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Invoke the Task in Parallel
# MAGIC
# MAGIC Use the `process_chunk` task to process multiple chunks of data concurrently.
# MAGIC

# COMMAND ----------

import random

def generate_lists(X, Y):
    lists = {}
    for i in range(X):
        variable_name = f'list_{i+1}'
        lists[variable_name] = [random.randint(1, 100) for _ in range(Y)]
    return lists

# Example usage
X = 100  # Number of lists
Y = 100  # Number of elements in each list
generated_lists = generate_lists(X, Y)
for name, lst in generated_lists.items():
    print(f"{name}: {lst}")
    break

# COMMAND ----------

# Dispatch tasks for each data chunk in parallel
result_ids = [normalize_list.remote(v) for k,v in generated_lists.items()]

# Gather results
results = ray.get(result_ids)
print("Processed data chunks:", results[0][:5])

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**: We create a list of data chunks and dispatch `process_chunk` as a task for each chunk. Using `ray.get`, we gather the results, demonstrating how tasks enable parallel, distributed data processing.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC - [Ray Documentation - Tasks](https://docs.ray.io/en/latest/walkthrough.html#remote-functions-and-actors)
# MAGIC - [Ray Remote Function Basics](https://docs.ray.io/en/latest/walkthrough.html#tasks)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Define a Ray Actor Class
# MAGIC
# MAGIC Let's define a `DataProcessor` class with a method to process healthcare data. We will use `@ray.remote` to create a remote class that allows multiple instances (actors) to run in parallel.
# MAGIC

# COMMAND ----------

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

@ray.remote
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process_data_standard(self):
        scaler = StandardScaler()
        data_array = np.array(self.data).reshape(-1, 1)
        processed_data = scaler.fit_transform(data_array).flatten().tolist()
        return processed_data

    def process_data_minmax(self):
        scaler = MinMaxScaler()
        data_array = np.array(self.data).reshape(-1, 1)
        processed_data = scaler.fit_transform(data_array).flatten().tolist()
        return processed_data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Create and Use Remote Class Instances (Actors)
# MAGIC
# MAGIC We can now create multiple instances of `DataProcessor` in parallel and invoke the `process_data` method on each. 
# MAGIC

# COMMAND ----------

# Sample data for processing (e.g., patient records or health metrics)
sample_data_1 = [100, 200, 300]
sample_data_2 = [400, 500, 600]

# Initialize two remote actors with different data
processor_1 = DataProcessor.remote(sample_data_1)
processor_2 = DataProcessor.remote(sample_data_2)

# Run process_data on each actor and retrieve the results
results = ray.get([processor_1.process_data_standard.remote(), 
                   processor_2.process_data_standard.remote(), 
                   processor_1.process_data_minmax.remote(), 
                   processor_2.process_data_minmax.remote()])

# Print results with function names
print("Standard Scaler - Processor 1:", results[0])
print("Standard Scaler - Processor 2:", results[1])
print("MinMax Scaler - Processor 1:", results[2])
print("MinMax Scaler - Processor 2:", results[3])

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**: By calling `.remote()` on `DataProcessor`, we create actor instances (`processor_1` and `processor_2`) that each handle a different dataset. Using `ray.get()`, we collect the processed data from each actor, simulating parallel data handling, which is common in healthcare data processing scenarios.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC - [Ray Documentation - Remote Functions and Classes](https://docs.ray.io/en/latest/walkthrough.html#remote-functions-and-actors)
# MAGIC - [Ray Actor Programming](https://docs.ray.io/en/latest/actors.html)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3: Simple Function Comparison
# MAGIC
# MAGIC ### Objective
# MAGIC Compare the runtime of a simple function executed sequentially versus parallelized using Ray. This demonstrates how Ray can speed up independent tasks that can be processed concurrently.
# MAGIC
# MAGIC ### Example Function
# MAGIC We’ll use a function that simulates a computational task.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Define a Simple Function
# MAGIC
# MAGIC Define a function to simulate a computational task.
# MAGIC

# COMMAND ----------

import time

def calculate_score(data):
    # Simulate a computation (e.g., calculating a score from health metrics)
    time.sleep(1)  # Sleep to simulate a time-intensive task
    return sum(data) / len(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Sequential Execution
# MAGIC
# MAGIC Run this function sequentially to process multiple datasets and time its execution.
# MAGIC

# COMMAND ----------

# Sample data sets
data_1 = [50, 60, 70]
data_2 = [80, 90, 100]
data_3 = [65, 75, 85]

# Sequential execution
start_time = time.time()
results_sequential = [calculate_score(data) for data in [data_1, data_2, data_3]]
end_time = time.time()

print("Sequential results:", results_sequential)
print("Sequential execution time:", end_time - start_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Parallel Execution with Ray
# MAGIC
# MAGIC Convert the function to a Ray task using `@ray.remote`, and then execute it in parallel.
# MAGIC

# COMMAND ----------

@ray.remote
def calculate_score_ray(data):
    time.sleep(1)  # Simulate a time-intensive task
    return sum(data) / len(data)

# COMMAND ----------

# Parallel execution
start_time = time.time()
results_parallel = ray.get([calculate_score_ray.remote(data) for data in [data_1, data_2, data_3]])
end_time = time.time()

print("Parallel results:", results_parallel)
print("Parallel execution time:", end_time - start_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**: The sequential execution processes each dataset one at a time, while the parallel execution uses Ray tasks to process them concurrently, demonstrating a potential speed-up.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC - [Ray Remote Function Basics](https://docs.ray.io/en/latest/walkthrough.html#tasks)
# MAGIC - [Ray Documentation - Task Parallelism](https://docs.ray.io/en/latest/walkthrough.html#remote-functions-and-actors)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Distributed Scikit-learn / Joblib
# MAGIC Ray supports running distributed scikit-learn programs by implementing a Ray backend for joblib using Ray Actors instead of local processes. This makes it easy to scale existing applications that use scikit-learn from a single node to a cluster.

# COMMAND ----------

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

digits = load_digits()

param_space = {
    'n_estimators': [100, 200, 300],
    'learning_rate': np.logspace(-3, 0, 30),
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
}
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
search = RandomizedSearchCV(model, param_space, cv=5, n_iter=100, verbose=10, n_jobs=-1)

# this method is using default scikit-learn API with a single node mode
search.fit(digits.data, digits.target)

# COMMAND ----------

# this cell is using joblib to run scikit-learn in parallel
import joblib
from ray.util.joblib import register_ray

register_ray()
with joblib.parallel_backend('ray'):
    search.fit(digits.data, digits.target)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4: Time Series Forecasting Comparison (Ray and MLflow)
# MAGIC
# MAGIC ### Objective
# MAGIC Compare sequential versus parallel time series forecasting using Ray and log the results to MLflow. This integration allows tracking of model parameters and forecasts, making it easier to analyze and compare results.
# MAGIC
# MAGIC ### Example Model
# MAGIC We’ll use an ARIMA model for time series forecasting on synthetic product sales data, logging results in MLflow for each forecast.
# MAGIC

# COMMAND ----------

# Import necessary libraries
import ray
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import time
import mlflow

# Initialize Ray
ray.init(ignore_reinit_error=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create Synthetic Time Series Data for Products
# MAGIC
# MAGIC Generate synthetic time series data for different products to simulate historical sales.
# MAGIC

# COMMAND ----------

# Generate synthetic data for 5 products over 100 time periods
np.random.seed(42)
time_periods = 100
num_products = 5
product_data = {
    f"product_{i}": np.cumsum(np.random.randn(time_periods) + 0.1 * i) for i in range(num_products)
}

# Convert to DataFrame for easier handling
df = pd.DataFrame(product_data)
print(df.head())

# COMMAND ----------

def forecast_sales(data, order=(1, 1, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)  # Forecast one period ahead
    return forecast[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Define Time Series Forecasting Function with MLflow Tracking
# MAGIC
# MAGIC Define a function that fits an ARIMA model to a product’s time series data, forecasts the next period, and logs results to MLflow.
# MAGIC

# COMMAND ----------

def forecast_sales_with_mlflow(data, product_name, order=(1, 1, 1)):
    # with mlflow.start_run(run_name=f"Forecast_{product_name}"):
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        
        # Log model parameters and forecast result
        mlflow.log_param("order", order)
        mlflow.log_metric("forecast", forecast)
        
        return forecast

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Sequential Forecasting with MLflow
# MAGIC
# MAGIC Run the time series forecasting for each product sequentially and log results to MLflow.
# MAGIC

# COMMAND ----------

# Sequential forecasting with MLflow logging
start_time = time.time()
forecasts_sequential = {
    product: forecast_sales_with_mlflow(df[product].values, product) for product in df.columns
}
end_time = time.time()

print("Sequential forecasts:", forecasts_sequential)
print("Sequential forecasting time:", end_time - start_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Parallel Forecasting with Ray and MLflow
# MAGIC
# MAGIC Convert the forecasting function into a Ray task to enable parallel execution with MLflow tracking.
# MAGIC

# COMMAND ----------

import ray
import mlflow
from statsmodels.tsa.arima.model import ARIMA
import time
from mlflow.utils.databricks_utils import get_databricks_env_vars
mlflow_db_creds = get_databricks_env_vars("databricks")

username = "guanyu.chen@databricks.com" # Username path
experiment_name = f"/Users/{username}/mlflow_ray_test"

mlflow.set_experiment(experiment_name)

@ray.remote
def forecast_sales_parallel_with_mlflow(data, product_name, run_id, order=(1, 1, 1)):
    import os
    # Set the MLflow credentials within the Ray task
    os.environ.update(mlflow_db_creds)
    # Set the active MLflow experiment within each Ray task
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_id=run_id, nested=True):  # Enable nested runs for parallel tasks
                
        # Fit ARIMA model
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=1)[0]
        
        # Log metrics
        mlflow.log_metric("forecast_value", forecast)
        
        return forecast

# COMMAND ----------

# Example usage
start_time = time.time()

# Start a parent MLflow run to group all parallel tasks
with mlflow.start_run(run_name="Parallel Sales Forecasting") as run:
    forecasts_parallel = ray.get([
        forecast_sales_parallel_with_mlflow.remote(df[product].values, product, run.info.run_id)
        for product in df.columns
    ])

end_time = time.time()

print("Parallel forecasts:", dict(zip(df.columns, forecasts_parallel)))
print("Parallel forecasting time:", end_time - start_time, "seconds")


# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**: The sequential approach processes each data split one at a time, while the parallel approach runs the training task concurrently on all splits. This comparison showcases the potential time savings from parallelizing model training with Ray.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC - [Ray Documentation - Parallelism in Machine Learning](https://docs.ray.io/en/latest/ray-core/walkthrough.html#tasks)
# MAGIC - [Ray and scikit-learn Integration](https://docs.ray.io/en/latest/ray-more-libraries/ray-sklearn.html)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5: Hyperparameter Tuning Comparison
# MAGIC
# MAGIC ### Objective
# MAGIC Compare sequential versus parallel hyperparameter tuning for a simple ML model using Ray’s hyperparameter tuning capabilities. This is particularly beneficial in healthcare applications where optimizing model parameters for different patient datasets can yield better predictions.
# MAGIC
# MAGIC ### Example
# MAGIC We’ll use a basic decision tree regressor and tune its `max_depth` hyperparameter to find the best configuration.
# MAGIC

# COMMAND ----------

# Import additional libraries for hyperparameter tuning
import ray
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import numpy as np
import time

# Initialize Ray
# ray.init(ignore_reinit_error=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create Sample Data
# MAGIC
# MAGIC Generate synthetic healthcare data for regression analysis.
# MAGIC

# COMMAND ----------

# Generate synthetic data
data, target = make_regression(n_samples=50000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Traditional Sklearn Hyperparameter Tuning
# MAGIC
# MAGIC We’ll define a function to train a model with a specific hyperparameter (in this case, `max_depth`, etc) and return the MSE.
# MAGIC

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [2, 4, 6, 8, 10, 12, 14],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
    'max_features': ['sqrt', 'log2']
}

# Initialize the model
model = DecisionTreeRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

start_time = time.time()
# Fit the model
grid_search.fit(X_train, y_train)
end_time = time.time()

# Get the best parameters and the corresponding MSE
best_params = grid_search.best_params_
best_mse = -grid_search.best_score_

print("Best MSEs:", best_mse)
print("Best parameters:", best_params)
print("Sequential tuning time:", end_time - start_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Parallel Hyperparameter Tuning with Ray Tune
# MAGIC
# MAGIC Convert the tuning function to a Ray task and run it in parallel.
# MAGIC

# COMMAND ----------

# this cell would take about 3 min to run without setting up Ray clusters on Databricks
# once Ray clusters are set up on Databricks, this cell would take about 40s to run

from ray import tune
import numpy as np
import ray

def objective(config): 
    score = config["a"] ** 2 + config["b"] * np.sin(config["c"]) + np.log(np.abs(config["d"]) + 1) * config["e"] ** 2
    return {"score": score}

search_space = { 
    "a": tune.grid_search([0.001, 0.01, 0.1]),
    "b": tune.choice([1, 2, 3, 4, 5, ]),
    "c": tune.uniform(0, 4 * np.pi),
    "d": tune.loguniform(1e-5, 1e3),
    "e": tune.quniform(1, 200, 1)
}

tuner = tune.Tuner(
    objective, 
    param_space=search_space,
    tune_config=tune.TuneConfig(num_samples=100)
)  

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)

# COMMAND ----------

import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.search.basic_variant import BasicVariantGenerator

# Define the training function
def train_model(config):
    # Initialize the model with hyperparameters from config
    model = DecisionTreeRegressor(
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"]
    )
    # Fit the model
    model.fit(X_train, y_train)
    # Predict on the test set
    preds = model.predict(X_test)
    # Calculate mean squared error
    mse = mean_squared_error(y_test, preds)
    # Report the MSE to Ray Tune
    return {"mse": mse}

# Define the search space
param_space = {
"max_depth": tune.choice([2, 4, 6, 8, 10, 12, 14]),
"min_samples_split": tune.choice([2, 5, 10, 15, 20]),
"min_samples_leaf": tune.choice([1, 2, 4, 6, 8, 10]),
"max_features": tune.choice(['sqrt', 'log2'])
}

# hyperopt_search = HyperOptSearch(space, metric="mean_accuracy", mode="max")

tuner = Tuner(
    train_model,
    param_space=param_space,
    tune_config=TuneConfig(mode='min',
                        search_alg='variant_generator', #search_alg=hyperopt_search, also can use optuna as search_alg
                        num_samples=210, # Number of times to sample from the hyperparameter space.
                        )
)

results = tuner.fit()

# Start the hyperparameter tuning
start_time = time.time()
results = tuner.fit()
end_time = time.time()

# Retrieve the best result
best_result = results.get_best_result(metric="mse", mode="min")
best_params = best_result.config
best_mse = best_result.metrics["mse"]

print("Best MSE:", best_mse)
print("Best parameters:", best_params)
print("Parallel tuning time:", end_time - start_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**: By parallelizing the hyperparameter tuning, each value is tried independently by a different Ray worker, enabling concurrent evaluation.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC - [Ray Documentation - Hyperparameter Tuning](https://docs.ray.io/en/latest/tune/index.html)
# MAGIC - [Ray Tune for Distributed Hyperparameter Search](https://docs.ray.io/en/latest/tune/key-concepts.html)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6: Neural Network Comparison
# MAGIC
# MAGIC ### Objective
# MAGIC Compare the training of a simple neural network model in a sequential versus parallelized manner using Ray. This section demonstrates how Ray can be used to speed up neural network training by distributing training tasks across different data subsets or configurations.
# MAGIC
# MAGIC ### Example Model
# MAGIC We’ll use a basic feedforward neural network to predict values in a regression task, which could be applied to healthcare metrics.
# MAGIC

# COMMAND ----------

# Import necessary libraries
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

# Initialize Ray
# ray.init(ignore_reinit_error=True)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Define the Neural Network Model
# MAGIC
# MAGIC Define a simple feedforward neural network using PyTorch.
# MAGIC

# COMMAND ----------

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create Sample Data and Preprocess
# MAGIC
# MAGIC Generate synthetic healthcare data and preprocess it for neural network training.
# MAGIC

# COMMAND ----------

# Generate synthetic data
data, target = make_regression(n_samples=5000, n_features=10, noise=0.1, random_state=42)
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Convert data to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Define Training Function
# MAGIC
# MAGIC Define a function to train the neural network.
# MAGIC

# COMMAND ----------

def train_neural_network(X_train, y_train):
    model = SimpleNN(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return loss.item()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Neural Network Training
# MAGIC
# MAGIC Train the neural network model sequentially for a fixed number of epochs and measure the time.
# MAGIC

# COMMAND ----------

# Sequential training
start_time = time.time()
loss_sequential = train_neural_network(X_train_tensor, y_train_tensor)
end_time = time.time()

print("Sequential training loss:", loss_sequential)
print("Sequential training time:", end_time - start_time, "seconds")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Parallel Neural Network Training with Ray
# MAGIC
# MAGIC Convert the training function to a Ray task to enable parallel execution across multiple subsets or configurations.
# MAGIC

# COMMAND ----------

# Define the training function
def train_func(config):
    # Create a dataset from the provided tensors
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32)

    # Create model, loss function, and optimizer
    model = SimpleNN(input_size=X_train_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        train.report({"loss": loss.item()})

# COMMAND ----------

# Create a TorchTrainer
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(
        num_workers=40, # 8 cores per node, 5 nodes including driver
        resources_per_worker={"CPU":0.5, "GPU":0}, # 0 GPUs per worker, 0.5 CPU per worker
        use_gpu=False)
)

# Parallel training
start_time = time.time()
# Start distributed training
result = trainer.fit()
end_time = time.time()

print("Parallel training losses:", result.metrics['loss'])
print("Parallel training time:", end_time - start_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**: The sequential training processes the full dataset, while parallel training distributes the data across multiple tasks, each trained independently. This approach illustrates how Ray can parallelize neural network training for multiple configurations or subsets.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC - [Ray Documentation - Neural Network Training](https://docs.ray.io/en/latest/ray-core/walkthrough.html#tasks)
# MAGIC - [Ray with PyTorch for Distributed Training](https://docs.ray.io/en/latest/train/train.html)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7: Comparison with Databricks Foundational Model APIs
# MAGIC
# MAGIC ### Objective
# MAGIC Compare the performance of Databricks Foundational Model API in sequential and parallel scenarios. Using Databricks API can provide scalable model serving for production use cases.
# MAGIC

# COMMAND ----------

# Import necessary libraries
import ray
import requests
import time

# Initialize Ray
ray.init(ignore_reinit_error=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Configure Databricks Model Serving API Endpoint
# MAGIC
# MAGIC Define the API endpoint and headers for authenticating requests to Databricks.

# COMMAND ----------

from openai import OpenAI
import os

access_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host_url = f"https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get('browserHostName').get()}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Define a Function for Databricks API Inference
# MAGIC
# MAGIC Create a function that sends a prompt to the Databricks Model Serving API and retrieves the response.
# MAGIC

# COMMAND ----------

def dbrx_generate_response(query: str):
   client = OpenAI(
     api_key=access_token,
     base_url=f"{host_url}/serving-endpoints"
   )

   chat_completion = client.chat.completions.create(
     messages=[
       {
         "role": "system",
         "content": "You are an AI assistant"
       },
       {
         "role": "user",
         "content": query
       },
     ],
     model="databricks-dbrx-instruct",
     max_tokens=256
   )
   return chat_completion.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Sequential Execution with Databricks API
# MAGIC
# MAGIC Run inference sequentially using Databricks API and measure execution time.
# MAGIC

# COMMAND ----------

# Sample healthcare-related prompts
prompts = [
    "What are the symptoms of diabetes?",
    "Explain the treatment options for high blood pressure.",
    "What should a patient know about cholesterol management?",
    "Describe the causes of heart disease.",
    "How does diet impact mental health?"
]

# Sequential inference using Databricks API
start_time = time.time()
responses_sequential = [dbrx_generate_response(prompt) for prompt in prompts]
end_time = time.time()

print("Sequential Databricks API execution time:", end_time - start_time, "seconds")
print("Sequential Databricks API responses:")
for i, response in enumerate(responses_sequential):
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Response {i+1}: {response}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Parallel Execution with Ray and Databricks API
# MAGIC
# MAGIC Convert the function to a Ray remote function and run inference in parallel using Databricks API.
# MAGIC

# COMMAND ----------

@ray.remote
def dbrx_generate_response_parallel(query: str):
   client = OpenAI(
     api_key=access_token,
     base_url=f"{host_url}/serving-endpoints"
   )

   chat_completion = client.chat.completions.create(
     messages=[
       {
         "role": "system",
         "content": "You are an AI assistant"
       },
       {
         "role": "user",
         "content": query
       },
     ],
     model="databricks-dbrx-instruct",
     max_tokens=256
   )
   return chat_completion.choices[0].message.content

# COMMAND ----------

# Parallel inference with Databricks API
start_time = time.time()
responses_parallel = ray.get([dbrx_generate_response_parallel.remote(prompt) for prompt in prompts])
end_time = time.time()

print("Parallel Databricks API execution time:", end_time - start_time, "seconds")
print("Parallel Databricks API responses:")
for i, response in enumerate(responses_parallel):
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Response {i+1}: {response}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparison of Execution Times
# MAGIC
# MAGIC This section compares the execution times of using a locally hosted model (like Hugging Face) and Databricks Foundational Model API in both sequential and parallel scenarios.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC - [Databricks Model Serving Documentation](https://docs.databricks.com/applications/mlflow/model-serving.html)
# MAGIC - [Ray Documentation - Parallel Task Processing](https://docs.ray.io/en/latest/ray-core/walkthrough.html#tasks)
# MAGIC - [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
# MAGIC

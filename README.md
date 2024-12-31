# a-synthetic-dataset-with-1-million-samples-and-10-features-using-make_regression-from-sklearn
This project demonstrates the implementation of mini-batch Stochastic Gradient Descent (SGD) for training a linear regression model on a large synthetic dataset. The dataset consists of 1 million samples with 10 features each, generated using sklearn.datasets.make_regression. The goal is to optimize the weights and bias of the model by minimizing the loss (Mean Squared Error) using SGD.

Key Features:

Mini-batch Stochastic Gradient Descent for efficient training on large datasets.

Training on synthetic data: 1 million samples and 10 features.

Real-time progress tracking: Displays time elapsed every 100 iterations.

Randomized dataset shuffling: Ensures that each mini-batch is different in each iteration.


Requirements

Python 3.x

numpy (for numerical computations)

pandas (for data handling, though not directly used in the current implementation)

scikit-learn (for dataset generation)


You can install the necessary dependencies with the following:

pip install numpy pandas scikit-learn

How to Run

1. Clone or download the repository containing the code.


2. Install the required dependencies by running:

pip install numpy pandas scikit-learn


3. Run the Python script:

python linear_regression_sgd.py


4. Observe the output:

The script will print progress messages every 100 iterations, including the time elapsed since the start of training.

Once training is complete, the final weights and bias values will be displayed along with the total training time.




Explanation of the Code

Dataset Generation

The dataset is generated using make_regression from sklearn.datasets, which creates a synthetic regression dataset with:

n_samples = 1,000,000 (1 million samples)

n_features = 10 (10 features)

noise = 0.1 to introduce some random noise to the target values y.



Parameters and Initialization

The model is initialized with random weights and bias:

weights initialized using np.random.randn().

bias initialized using np.random.randn().


The following hyperparameters are used:

learning_rate = 0.01: The learning rate for gradient descent.

n_iterations = 1000: The number of iterations for training.

batch_size = 1000: The mini-batch size used for each gradient descent update.



Mini-batch Stochastic Gradient Descent (SGD)

The core of the implementation is the SGD algorithm. It works by:

1. Shuffling the dataset: This ensures that the model doesn't overfit to the order of data and is trained on random mini-batches.


2. Processing mini-batches: The dataset is divided into smaller batches (of size batch_size). The gradient is computed and the weights and bias are updated after each mini-batch.


3. Updating model parameters: After each mini-batch, the weights and bias are updated by computing the gradients of the Mean Squared Error (MSE) loss function with respect to the weights and bias.



Helper Function: sgd_step

The sgd_step function performs the following:

1. Calculates predictions: Using the current model parameters (weights and bias).


2. Computes gradients: The gradient of the loss function with respect to the weights and bias.


3. Updates weights and bias: Using the computed gradients and the learning rate.



Progress Tracking

Every 100 iterations, the script prints the number of iterations and the time elapsed since the start of training. This helps to track the progress, especially for long-running training jobs.


Final Output

After 1000 iterations, the final model parameters (weights and bias) are printed.

The total time taken for training is also displayed at the end.
Final weights: [ 0.29847395 -0.03187434  0.21261402  1.07507092  0.42346251 -0.20842975 -0.35707856 -0.42909206  0.03934042 -0.17236462]
Final bias: -0.11305031733769068
Training completed in 52.74 seconds

This output provides information on the model's final parameters and the time it took to train the model.

Improvements & Future Work

1. Add Loss Calculation: Implement the calculation and printing of the Mean Squared Error (MSE) loss for each mini-batch or iteration to help monitor training progress.


2. Early Stopping: Implement early stopping to terminate the training early if the model converges or if the validation loss stops improving.


3. Hyperparameter Tuning: Explore different learning rates, batch sizes, and other hyperparameters to optimize model performance.


4. Distributed Training: For even larger datasets, consider using distributed computing frameworks like Dask or PySpark.

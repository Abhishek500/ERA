# S5

This project focuses on getting familiar with github and exploring modularity in pytorch.

## Project Structure
 After fixing the code , I split the code into following sections, for better organization and maintainability:
- `train.py`: This file provides functions for training and evaluating a model. The 'train' function trains the model by iterating over batches of data, performing forward and backward passes, and updating the model's parameters. The 'test' function evaluates the trained model on a test set and computes the average test loss and accuracy.
- `utils.py`: This file contains utility function (getting count of correct predictions)  and helper method to plot 12 images from the batch.
- `data_loader.py`: This file handles the data loading and preprocessing tasks. It includes functions to load the dataset, perform any required transformations, and create data loaders for efficient batch processing during training.
- `model.py`: This file defines the architecture of the neural network model used in the project. It includes the model class, layers. Separating the model code allows for easy experimentation and model swapping.
- `Session_5.ipynb`: This Jupyter Notebook provides an interactive environment for exploring the project. It showcases examples of using the code from the different files and demonstrates the project's functionality. It serves as a practical guide and can be used for further experimentation or as a starting point for future work.

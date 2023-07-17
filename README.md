<h1 style="text-align: center;"><b> Sentiment Analysis using GRU, LSTM and LSTM Bidirectional </b></h1>

This is a work we've done in the context of a practical tutorial at university, the goal was to classify sentiments from texts through different neural networks architectures. A lot more details are present in the notebook
This repository contains Python code for building and training different Recurrent Neural Networks (RNNs) models to perform sentiment analysis on the IMDB movie reviews dataset.

## Dataset
The dataset used for sentiment analysis is the IMDB movie reviews dataset. The data is automatically downloaded and preprocessed using TensorFlow's `text_dataset_from_directory` function.

## Data Preprocessing
Before feeding the data to the models, text data is preprocessed using a custom standardization function. The text is converted to lowercase, HTML tags are removed, and punctuation marks are stripped from the text.

## Model Architectures

### RNN (Recurrent Neural Network)
The first model is a simple RNN with Gated Recurrent Unit (GRU) cells. The architecture is as follows:

1. Embedding layer with 10000 maximum features and an input sequence length of 250.
2. SpatialDropout1D layer with a dropout rate of 0.4.
3. GRU layer with 32 units and a dropout rate of 0.05 in the input and recurrent connections.
4. Dense layer with a sigmoid activation function for binary classification.

### LSTM Bidirectional
The second model is a Bidirectional Long Short-Term Memory (LSTM) network, which processes the input sequence in both forward and backward directions. The architecture is as follows:

1. Embedding layer with 10000 maximum features and an input sequence length of 250.
2. SpatialDropout1D layer with a dropout rate of 0.4.
3. Bidirectional LSTM layer with 32 units and a dropout rate of 0.05 in the input and recurrent connections.
4. Dense layer with a sigmoid activation function for binary classification.

### LSTM (Long Short-Term Memory)
The third model is a simple LSTM network. The architecture is as follows:

1. Embedding layer with 10000 maximum features and an input sequence length of 250.
2. SpatialDropout1D layer with a dropout rate of 0.3.
3. LSTM layer with 128 units and a dropout rate of 0.5 in the input and recurrent connections.
4. Dense layer with a sigmoid activation function for binary classification.

## Model Training
All models are trained using the BinaryCrossentropy loss function and the Adam optimizer. The training is performed on the preprocessed text data, and validation is done on a separate validation dataset.

## Performance Visualization
After training, the models' performance is visualized by plotting the training and validation loss over the epochs.

## Requirements
- TensorFlow 
- matplotlib 

## How to Use
1. Download and preprocess the IMDB dataset using the provided code.
2. Run the RNN, LSTM Bidirectional, and LSTM models training code separately.
3. Visualize the training and validation loss for each model using the provided code.

## Contribution
Feel free to fork, modify, and use the code for your own sentiment analysis projects!

For any questions or issues, please open an issue in this repository.

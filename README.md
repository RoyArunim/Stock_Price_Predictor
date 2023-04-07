# Stock Price Predictor
Predicts the closing price of each stock at the end of the day with existing data.
Here we use LSTM to preditct the closing price of the stock prices mentioned in the IXIC.csv file. To prevent overfitting we have specified Dropout regularization to be 20% of the 
return sequences, such that 20 % of the sequences will be probabilistically excluded from the activation sequences.

After pre-processing of data, while building the LSTM model, following steps are followed:
Build the LSTM model 
 We add the LSTM layer and later add a few Dropout layers to prevent overfitting. 
 We add the LSTM layer with the following arguments:
 1. 50 units which is the dimensionality of the output space
 2. return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence
 3. input_shape as the shape of our training set.

 When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped.
 Thereafter, we add the Dense layer that specifies the output of 1 unit
 After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error.
 Next, we fit the model to run on 100 epochs with a batch size of 32

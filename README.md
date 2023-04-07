# Stock_Price_Predictor
Predicts the closing price of each stock at the end of the day with existing data.
Here we use LSTM to preditct the closing price of the stock prices mentioned in the IXIC.csv file. To prevent overfitting we have specified Dropout regularization to be 20% of the 
return sequences, such that 20 % of the sequences will be probabilistically excluded from the activation sequences.


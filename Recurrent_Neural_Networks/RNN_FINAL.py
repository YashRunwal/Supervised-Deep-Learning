# Recurrent Neural Network - Predicting stock price of Google

# Data Preprocessing 

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

''' STEP 1'''
# Import training set
train_dataset = pd.read_csv('Google_Stock_Price_Train.csv')

# Select the required columns from the csv file

# This is the training set on which RNN is trained (# 2 is excluded...numpy array of one column..column 1)
training_set = train_dataset.iloc[:, 1:2].values  

# Feature Scaling (Standardisation and Normalisation)
sc = MinMaxScaler(feature_range=(0,1), copy=True)
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure that RNN should remember - Number of Time Steps
# Time Steps - 60 with 1 Output
x_train_is_input  = []
y_train_is_output = []

for i in range(60, len(training_set)):
    x_train_is_input.append(training_set_scaled[i-60:i, 0])
    y_train_is_output.append(training_set_scaled[i, 0])
    

# Convert x_train_input and y_train_output to numpy arrays, so that we can feed into RNN
x_train_is_input, y_train_is_output = np.array(x_train_is_input), np.array(y_train_is_output)

# Reshaping Data
x_train_is_input = np.reshape(x_train_is_input, (x_train_is_input.shape[0], x_train_is_input.shape[1], 1))


''' STEP 2 '''
# Build the RNN

# Initialize the RNN
regressor = Sequential()

# Add First LSTM Layer and Dropout Regularization (To Avoid Overfitting)
'''
units - number of Neurons
return_sequences = True because more LSTM layers will be added
'''
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train_is_input.shape[1], 1)))
regressor.add(Dropout(rate = 0.2)) # 0.2  is 20 percent. So 20 percent of 50 is 10 neurons. 10 neurons will be dropped.

# Additional LSTM Layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(rate = 0.2))

# Output Layer
regressor.add(Dense(units = 1))


''' STEP 3 '''
# Making the predictions
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit the RNN to the training set
regressor.fit(x = x_train_is_input, y = y_train_is_output, epochs = 75, batch_size = 32)

# Gettting the real stock price of Google in 2017
test_dataset = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_dataset.iloc[:, 1:2].values

# Gettting the predicted stock price of Google in 2017
dataset_concentenate = pd.concat((train_dataset['Open'], test_dataset['Open']), axis = 0)
inputs = dataset_concentenate[len(dataset_concentenate) - len(test_dataset) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs) # transform is used to get the previous scaling on which training set was scaled

x_test  = []
for i in range(60, test_dataset.shape[0] + x_train_is_input.shape[1]):
    x_test.append(inputs[i-60:i, 0])
    
# Convert x_train_input and y_train_output to numpy arrays, so that we can feed into RNN
x_test = np.array(x_test)

# Reshaping Data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualize
plt.plot(real_stock_price, color = 'blue', label = 'Real Stock Price of Jan 2017')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Stock Price of Jan 2017')
plt.title("Comparison of Real and Predicted Stock Prices for Jan 2017")
plt.xlabel("Days of Jan 2017")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

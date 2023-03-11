# The Machine Learning Workflow
## Data Collection 
We collected data via web scraping and the data were pre processed further to filter only ‘Company Name’, ‘Closing Price’, ‘Date’, and ‘Volume’.

## Linear Interpolation to Fill Missing Data
Data for the days the stock market stays off is not available. Which means there are gaps in our time-series data. We used linear interpolation to fill the gaps. This preprocessing alone improves the results (decreases the error) by 10%. 

## CNN-LSTM Prediction Model
A convolutional neural network (CNN) is a type of neural network developed for working with two-dimensional image data. The CNN can be very effective at automatically extracting and learning features from one-dimensional sequence data such as univariate time series data.[1]

Long Short Term Memory (LSTM) is a well known deep neural network which can deal with sequential time series data and capture future trends very well.

Here we have used CNN-LSTM hybrid model where CNN is used to interpret subsequences of input and the output of the CNN model is merged together and provided as a sequence to an LSTM model to interpret.

So instead of providing a whole sequence of data, we divide the data into subsequences and extract important features from the sequences using CNN, merge them and next provide this output as input to the LSTM model.

The steps -
1. First our data was split into input and output samples with 100 steps data as input 1 step data as output.
2. Then we have split our input samples into 25 sub-samples, each with 4 time steps.
3. The CNN now will interpret each subsequence of 4 time steps and provide a time series of interpretations of the subsequences to the LSTM model to process as input.
4. So the input sample of 100 days will be reshaped into (25, 4, 2) size where 25 is number of subsequences, 4 is timesteps and 2 is number of features (closing price and volume)
5. The convolution layer is followed by a max pooling layer that includes the most salient features. 
6. These structures are then flattened down to a single one-dimensional vector to be used as a single input time step to the LSTM layer. 

So in short, we are taking 100 days of data and using CNN we are extracting 25 features from subsequences and using the 25 sequence of features as input to the LSTM model.

## Model Train and Save
There are about 400 companies. We have trained our model separately for all companies and saved the model to future prediction.

## Prediction
We have predicted closing price and volume using this model. Like if a prediction of next 10 days is requested, then using the last 100 days data we have predicted 101th day and using 2nd day to 101th day data we have predicted 102th day data.


## Generate Buy-Sell Signal

Some technical indicators like On Balance Volume (OBV) and OBV EMA are calculated from volume. So we have used the formula to predict OBV and OBV EMA. 
And generated buy sell signals using these technical indicators. [2]
- If OBV > OBV_EMA Then Buy
- If OBV < OBV_EMA Then Sell
- Else Do nothing

## Assumptions
1. We have enough data points(at least 1000 data points) to train the model. 
2. DSE data is not stationary
3. Since CNN-LSTM predicted better than only LSTM (compared in terms of RMSE value) we assumed that for all other companies it will work well.
4. The configuration of the model was not changed company wise. Such as the same value of learning rate, batch size, subsequence size, epoch and loss function was used for all companies. We assumed that the same value will give a decent result, at least any particular trend.



## References
[1] Book: “Deep Learning for Time Series Forecasting” by Jason Brownlee

[2] Stock-trading-strategy-using-on-balance-volume-obv

# Stock-Price-Prediction
A model that predicts future stock prices of a company or currency using historical data with high accuracy. The program also visualizes the actual prices vs. the predicted prices and the training loss vs. the testing loss, etc. If cuda (GPU) is available, the program uses GPU to reduce training, validation and testing time.

Two architectures are used, LSTM and GRU. It is observed that GRU model performs better than LSTM model with the example data set (MSFT.csv). In this example, Microsoft stock prices between 1986 to 2022 are used.

Here are screenshots from the program output:

![MicroSlide](https://github.com/user-attachments/assets/ee1dc44a-d335-45d2-936b-c253846f6d32)

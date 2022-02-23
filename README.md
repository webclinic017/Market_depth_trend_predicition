# Market_depth_trend_predicition

This project is to determine whether the market-depth is useful to predict the next movement of the price
The project is intended to:
1. Save market_depth data from IB (TWS or IBGateway) as an SQL file
2. Use a tensorflow model to traind on the saved data and predict the trend of the asset price whether (1: up, 2: down, 0: stay at the same price) based on the top 5 best bids/asks prices in the market-depth book


This repository contains 2 python codes

# save_marketdepth_as_sql.py: 
Collect live tick by tick market depth from interactive broker and save it as a SQL (.db) file. If you have interactive brokers TWS installed in your computer with Futures data for level 2, you can download the market depth live and store it in your hard drive. You need to use port number 7496 in your TWS/IBGateway API settings. However, I put a sample of these data that you can use to try the model with.

# Market_depth_trend_prediction_model.py:
Train a model on the provided data using LSTM tensorflow model. 

I also included a trained model with accuracy/loss diagram to show how the model is performing

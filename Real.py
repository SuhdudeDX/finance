import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tickersymbols

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #long short term memory layers

root = Tk()
root.title('Stock Prices Predictor')

choice = StringVar()
Label(text = "Company", padx = 10, font = 30).grid(row = 0, sticky = W)
combo = ttk.Combobox(width = 15, font = 30, textvariable = choice)
combo["values"] = tickersymbols.COMPANY
combo.grid(row = 0, column = 1)

days = IntVar()
Label(text='Calculation Days', padx=10, font=100).grid(row=1, sticky = W)
mainroot = Entry(width=15, font=30, textvariable=days)
mainroot.grid(row=1, column=1)

start_year = StringVar()
Label(text='Start Year', padx=10, font=100).grid(row=2, sticky = W)
mainroot = Entry(width=15, font=30, textvariable=start_year)
mainroot.grid(row=2, column=1)

end_year = StringVar()
Label(text='End Year', padx=10, font=100).grid(row=3, sticky = W)
mainroot = Entry(width=15, font=30, textvariable=end_year)
mainroot.grid(row=3, column=1)

def enter():
    company = choice.get()
    prediction_days = days.get()
    starto = start_year.get()
    endo = end_year.get()
    starto_split = starto.split(', ')
    endo_split = endo.split(', ')
    for i in range(3):
        if i == 0:
            start_year1 = starto_split[0]
        elif i == 1:
            start_month1 = starto_split[1]
        else:
            start_day1 = starto_split[2]
    for i in range(3):
        if i == 0:
            end_year1 = endo_split[0]
        elif i == 1:
            end_month1 = endo_split[1]
        else:
            end_day1 = endo_split[2]
    print(dt.datetime.now())

    try:
        start = dt.datetime(int(start_year1), int(start_month1), int(start_day1))
        end = dt.datetime(int(end_year1), int(end_month1), int(end_day1))

        data = web.DataReader(company, 'yahoo', start, end)

        #Prepare Data
        scaler=MinMaxScaler(feature_range=(0,1))#fit prices into proper scale $10-$600 to 0-1
        scaled_data=scaler.fit_transform(data['Close'].values.reshape(-1,1))#predict from closing price and transform shape

        x_train=[]
        y_train=[]

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x-prediction_days:x, 0])#store data for training
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))#reshape to use with the neuro network model

        #Build the Model
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))#units=layers
        model.add(Dropout(0.2))#drop out is when randomly selected info are ignored during training
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1)) #prediction of the next closing value

        model.compile(optimizer='adam', loss='mean_squared_error')#calculations
        model.fit(x_train, y_train, epochs=25, batch_size=32)
        #epochs = model sees the same data 24 times/ batch sizes see 32 units at the same time

        '''Test the model accuracy on existing data'''

        #Load Test Data
        test_start=dt.datetime(int(end_year1), int(end_month1), int(end_day1))
        test_end=dt.datetime.now()

        test_data = web.DataReader(company, 'yahoo', test_start, test_end)
        actual_prices=test_data['Close'].values

        total_dataset=pd.concat((data['Close'], test_data['Close']), axis=0)#not scaled

        model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values#model sees input
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        # Make Predictions on Test Data
        x_test=[]

        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x-prediction_days:x, 0])

        x_test=np.array(x_test)
        x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices=model.predict(x_test)
        predicted_prices=scaler.inverse_transform(predicted_prices)#inverse back to actualy predicted price

        # Plot the test predictions
        plt.plot(actual_prices, color = "black", label=f"Actual {company} Price")
        plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
        plt.title(f"{company} Share Price")
        plt.xlabel("Time")
        plt.ylabel(f"{company} Share Price in USD")
        plt.legend()
        plt.show()

        #Predict Next Day
        real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1), 0]]
        real_data = np.array(real_data)
        real_data=np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

        prediction=model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        messagebox.showinfo('Prediction', f'The Prediction Price for tomorrow is {prediction}')

    except:
        messagebox.showerror("Error","Input Error, Try again")


Button(text="ENTER", font = 30, width = 15, command = enter).grid(row = 5, column = 1, sticky = W)

root.mainloop()

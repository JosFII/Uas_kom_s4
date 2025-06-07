import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
import math, time
from sklearn.metrics import mean_squared_error
from prophet import Prophet
torch.classes.__path__ = [] 
st.markdown(
'''
<style>
    .stApp {
   background-color: white;
    }
 
       .stWrite,.stMarkdown,.stTextInput,h1, h2, h3, h4, h5, h6 {
            color: purple !important;
        }
</style>
''',
unsafe_allow_html=True
)

st.title("Machine Learning Basics: Supervised Learning")
st.markdown('dibuat oleh: Joseph F.H. (20234920002)')
st.header("Flowchart")
st.image('flw.jpg')

st.header("Data: Stock dari bank bca")
st.markdown('''
data yang digunakan merupakan data stock dari bank baca di indonesia yang diambil dengan package yfinance. 
data memiliki total 238 row dengan 5 column atau 238x5. 
data memiliki variabel variabel berikut: 
- date: tanggal dari data stock
- close: harga tutup dari stock pada hari tersebut
- high: harga tertinggi dari stock pada hari tersebut
- low: harga terendah dari stock pada hari tersebut
- open: harga buka dari stock pada hari tersebut
- volume: banyak stock yang dijual

stock bca dipilih karena bca merupakan salah satu bank besar di indonesia
''')
ticker='BBCA.JK'
bca = yf.download(ticker, start="2024-06-05", end="2025-06-06")



st.subheader("Data Exploration")
st.markdown('jadi berikut merupakan data stock bca:')
bca

st.markdown('''terus kita akan mengecek nilai na atau mising data pada data saham, untuk mengecek digunakan kode: \n
            st.write(bca.isna().sum()) \n
jadi berikut total nilai na di data:''')
st.write(bca.isna().sum())
st.markdown('jadi bisa dilihat bahwa tidak terdapat nilai na di saham bca.')

st.markdown('berikut grafik dari harga close saham bca')
bca[['Close']].plot(figsize=(15, 6))
plt.title('bca Stock close2')
plt.xlabel('Time')
plt.ylabel('bca Stock close2')
plt.legend()
plt.savefig('bca_pred.png')
st.pyplot(plt.gcf())


st.subheader("Feature Engineering")
st.markdown('''pertama kita akan membuat nilai saham close bca variabelnya sendiri:\n
            close=bca['Close']''')
close=bca['Close']
st.markdown('''untuk prediksi mengunakan LSTM, pertama data harus dinormalisasikan, 
untuk menormalisasikan data digunakan kode berikut:\n
        close2 = bca[['Close']]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        close2['Close'] = scaler.fit_transform(close2['Close'].values.reshape(-1,1)) ''')
close2 = bca[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
close2['Close'] = scaler.fit_transform(close2['Close'].values.reshape(-1,1))
st.write(close2)
st.markdown('bisa dilihat bahwa data nilai close sudah dinormalisasikan')
st.markdown('''
untuk prediksi dengan prophet kita perlu merubah column date dan column close menjadi ds dan y. ini dilakukan dengan kode
berikut: \n
        close.to_csv('bbca_historical_data.csv')
        clos3=pd.read_csv('bbca_historical_data.csv')
        clos3 = clos3.rename(columns = {"Date":"ds","BBCA.JK":"y"}) #renaming the columns of the dataset
        clos3.head()
''')


close.to_csv('bbca_historical_data.csv')
clos3=pd.read_csv('bbca_historical_data.csv')
clos3 = clos3.rename(columns = {"Date":"ds","BBCA.JK":"y"}) #renaming the columns of the dataset
st.write(clos3.head())
st.markdown('jadi bisa dilihat bahwa nama column nya sudah diganti menjadi ds dan y')

st.subheader("Prediction Stock Market: Arima")

st.latex('''
d=0: y_t= Y_t 
''')
st.latex('''
d=1: y_t= Y_t - Y_{t-1}
''')
st.latex('''
d=2: y_t= (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2}) = Y_t - 2Y_{t-1} - Y_{t-2}
''')
st.latex(r'''
\hat{y_t} = \mu + \phi_1 Y_{t-1} + ... + \phi_p Y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q}
''')
st.markdown('rumus diambil dari https://people.duke.edu/~rnau/411arim.htm')
st.markdown('''jadi pertama kita akan mebagi data menjadi train dan test data, test data akan diambil 30 karena kita mau memprediksikan 30 hari kedepan, 
berikut kode yang digunakan: \n
        close2s = close
        close2s = close2s.tz_localize(None)
        close2s.index = pd.DatetimeIndex(close2s.index).to_period(freq='B')

        train_end_date = close2s.index[-30]  # Use all but the last 30 data points for training
        train_data = close2s[:train_end_date]
        test_data = close2s[train_end_date:]''')
close2s = close
# Preprocessing
close2s = close2s.tz_localize(None)
close2s.index = pd.DatetimeIndex(close2s.index).to_period(freq='B')

# Split the data into training and testing sets
train_end_date = close2s.index[-30]  # Use all but the last 30 data points for training
train_data = close2s[:train_end_date]
test_data = close2s[train_end_date:]

st.markdown('''
untuk melakukan arima, daya menggunakan auto arima dari package pmdarima yang dapat menentukan nilai p d, dan q dari arima serta membuat model arima. 
berikut kode yang digunakan: \n
        model_auto = auto_arima(train_data, 
                   start_p=1, start_q=1,
                   max_p=5, max_q=5,
                   seasonal=False,
                   stepwise=True,
                   trace=True)

        best_order = model_auto.order
        st.write(f"Auto ARIMA found that the best order is {best_order}")
        st.write(model_auto.summary())

''')
# Fitting the ARIMA model on the training data
model_auto = auto_arima(train_data, 
                   start_p=1, start_q=1,
                   max_p=5, max_q=5,
                   seasonal=False,
                   stepwise=True,
                   trace=True)

best_order = model_auto.order
st.write(f"Auto ARIMA found that the best order is {best_order}")
st.write(model_auto.summary())
st.markdown('jadi diatas ditemukan bahwa nilai p, d, q terbaik adalah 2,1,0 dan diatas juga terdapat summary dari model arima yang dibuat.')

st.markdown('''
untuk forecasting dan membuat grafik hasil forecast digunakan kode berikut:
        forecast, conf_int = model_auto.predict(n_periods=len(test_data), return_conf_int=True)

        def plot_forecast_vs_actual(train_data, test_data, forecast, conf_int, ticker):
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    
            ax1.plot(train_data.index.to_timestamp(), train_data.values, label='Historical Data')
            ax1.plot(test_data.index.to_timestamp(), test_data.values, label='Actual Future Data', color='green')
            ax1.plot(test_data.index.to_timestamp(), forecast, color='red', label='Forecast')
            ax1.fill_between(test_data.index.to_timestamp(), 
                             conf_int[:, 0], 
                             conf_int[:, 1], 
                             color='pink', alpha=0.3, label='Confidence Interval')
            ax1.set_title(f'{ticker} Stock close2 - Full History and Forecast')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('close2')
            ax1.legend()
            ax1.grid(True)
    
    
            zoom_start = train_data.index[-50]  # Start zoom 50 periods before test data
            ax2.plot(train_data.loc[zoom_start:].index.to_timestamp(), 
                     train_data.loc[zoom_start:].values, label='Training Data')
            ax2.plot(test_data.index.to_timestamp(), test_data.values, label='Actual Future Data', color='green')
            ax2.plot(test_data.index.to_timestamp(), forecast, color='red', label='Forecast')
            ax2.fill_between(test_data.index.to_timestamp(), 
                             conf_int[:, 0], 
                             conf_int[:, 1], 
                             color='pink', alpha=0.3, label='Confidence Interval')
            ax2.set_title(f'{ticker} Stock close2 - Zoomed Forecast vs Actual')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('close2')
            ax2.legend()
            ax2.grid(True)
    
    
            ax2.set_xlim(zoom_start, test_data.index[-1])
    
            plt.tight_layout()
            st.pyplot(plt.gcf())
        plot_forecast_vs_actual(train_data, test_data, forecast, conf_int, ticker)
''')
# Forecasting
forecast, conf_int = model_auto.predict(n_periods=len(test_data), return_conf_int=True)

def plot_forecast_vs_actual(train_data, test_data, forecast, conf_int, ticker):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: All data
    ax1.plot(train_data.index.to_timestamp(), train_data.values, label='Historical Data')
    ax1.plot(test_data.index.to_timestamp(), test_data.values, label='Actual Future Data', color='green')
    ax1.plot(test_data.index.to_timestamp(), forecast, color='red', label='Forecast')
    ax1.fill_between(test_data.index.to_timestamp(), 
                     conf_int[:, 0], 
                     conf_int[:, 1], 
                     color='pink', alpha=0.3, label='Confidence Interval')
    ax1.set_title(f'{ticker} Stock close2 - Full History and Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('close2')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Zoomed in on prediction period
    zoom_start = train_data.index[-50]  # Start zoom 50 periods before test data
    ax2.plot(train_data.loc[zoom_start:].index.to_timestamp(), 
             train_data.loc[zoom_start:].values, label='Training Data')
    ax2.plot(test_data.index.to_timestamp(), test_data.values, label='Actual Future Data', color='green')
    ax2.plot(test_data.index.to_timestamp(), forecast, color='red', label='Forecast')
    ax2.fill_between(test_data.index.to_timestamp(), 
                     conf_int[:, 0], 
                     conf_int[:, 1], 
                     color='pink', alpha=0.3, label='Confidence Interval')
    ax2.set_title(f'{ticker} Stock close2 - Zoomed Forecast vs Actual')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('close2')
    ax2.legend()
    ax2.grid(True)
    
    # Set x-axis limits for the zoomed plot
    ax2.set_xlim(zoom_start, test_data.index[-1])
    
    plt.tight_layout()
    st.pyplot(plt.gcf())

# Plot the forecast vs actual data
plot_forecast_vs_actual(train_data, test_data, forecast, conf_int, ticker)
st.markdown('''
bisa dilihat diatas pada grafik bahwa nilai prediksi arima lurus saja sedangkan nilai sebenarnya naik dan turun diakhir, 
ini karena arima hanya memprediksikan tren, jadi menurut ariam stock bca memiliki garis tren yang lurus untuk 30 hari kedepan.
''')


st.markdown('''
untuk menghitung rmse dari prediksi arima digunakan kode berikut: \n
        model = ARIMA(bca['Close'], order=(2, 1, 0))
        model_fit = model.fit()
        model_fit.summary()
        forecast = model_fit.forecast(steps=30)
        trainScoreA = math.sqrt(mean_squared_error(test_data, forecast))
        st.write('Test Score: %.2f RMSE' % (trainScoreA))
''')
model = ARIMA(bca['Close'], order=(2, 1, 0))
model_fit = model.fit()
model_fit.summary()
forecast = model_fit.forecast(steps=30)
testScoreA = math.sqrt(mean_squared_error(test_data, forecast))
st.write('Test Score: %.2f RMSE' % (testScoreA))
st.write('jadi didapatkan rmse sebesar  %.2f, jadi bisa dibilang model masih kurang bagus dalam memprediksikan harga saham'% (testScoreA) )

st.subheader("Prediction Stock Market: LSTM")
st.image('Long_Short-Term_Memory.png')
st.markdown('gambar diambil dari https://en.wikipedia.org/wiki/Recurrent_neural_network')
st.markdown('''
pertama kita akan membagi data mrnjadi train dan test, disini kita akan mengambil 60 data terkahir untuk test data, 
digunakan kode berikut:
        def load_data(stock, look_back):
            data_raw = stock.values # convert to numpy array
            data = []
    
   
            for index in range(len(data_raw) - look_back): 
                data.append(data_raw[index: index + look_back])
    
            data = np.array(data);
            test_set_size = int(np.round(0.2*data.shape[0]));
            train_set_size = data.shape[0] - (test_set_size);
    
            x_train = data[:train_set_size,:-1,:]
            y_train = data[:train_set_size,-1,:]
    
            x_test = data[train_set_size:,:-1]
            y_test = data[train_set_size:,-1,:]
    
            return [x_train, y_train, x_test, y_test]

        look_back = 60 # choose sequence length
        x_train, y_train, x_test, y_test = load_data(close2, look_back)

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)
''')
def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

look_back = 60 # choose sequence length
x_train, y_train, x_test, y_test = load_data(close2, look_back)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

st.markdown('''
terus kita kana membuat model, untuk model digunakan parameter beriku:
- input_dim = 1
- hidden_dim = 32
- num_layers = 2
- output_dim = 1
- num_epochs = 100\n
berikut kode yang digunakan: \n
        input_dim = 1
        hidden_dim = 32
        num_layers = 2
        output_dim = 1
        num_epochs = 100

        class LSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super(LSTM, self).__init__()
        
                self.hidden_dim = hidden_dim

        
                self.num_layers = num_layers


                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
       
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()


                out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

 
                out = self.fc(out[:, -1, :]) 

                return out
    
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

        loss_fn = torch.nn.MSELoss()

        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        num_epochs = 100
        hist = np.zeros(num_epochs)


        seq_dim =look_back-1  
''')

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
hist = np.zeros(num_epochs)

# Number of steps to unroll
seq_dim =look_back-1  

st.markdown('''
terus kita perlu mentrain model kita terlebih dahulu:
        for t in range(num_epochs):

            y_train_pred = model(x_train)

            loss = loss_fn(y_train_pred, y_train)
            if t % 10 == 0 and t !=0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()
''')

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    #model.hidden = model.init_hidden()
    
    # Forward pass
    y_train_pred = model(x_train)

    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

st.markdown('''
untuk melakukan prediksi digunakan kode: \n
        y_test_pred=model(x_test)
''')

y_test_pred=model(x_test)

st.markdown('''
terus karena dari tadi digunakan data yang telah dinormalisasikan, kita perlu mengembalikan data ke bentuk awalnya, 
untuk melakukan ini digunakan kode berikut:\n
        y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
       y_train = scaler.inverse_transform(y_train.detach().numpy())
       y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
       y_test = scaler.inverse_transform(y_test.detach().numpy())
''')

y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

st.markdown('jadi berikut plot untuk nilai saham yang diprediksi dan nilai saham sebenarnya')

figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()

axes.plot(bca[len(bca)-len(y_test):].index, y_test, color = 'red', label = 'Real bca Stock close2')
axes.plot(bca[len(bca)-len(y_test):].index, y_test_pred, color = 'blue', label = 'Predicted bca Stock close2')
#axes.xticks(np.arange(0,394,50))
plt.title('bca Stock close2 Prediction')
plt.xlabel('Time')
plt.ylabel('bca Stock close2')
plt.legend()
plt.savefig('bcapred.png')
st.pyplot(plt.gcf())

st.markdown('''
jadi bisa dilihat dari grafik diatas bahwa nilai yang diprediksi dari lstm cukup dekat dengan nilai sebenarnya. jadi bisa digunakan untuk memprediksikan harga saham
''')

st.markdown('''
jadi untuk menghitung nilai rmse dari model lstm digunakan kode berikut:\n
        trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
        st.write('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
        st.write('Test Score: %.2f RMSE' % (testScore))
''')

trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
st.write('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
st.write('Test Score: %.2f RMSE' % (testScore))

st.write('jadi didapatkan rmse untuk training sebesar  %.2f, jadi bisa dibilang model masih kurang bagus dalam memprediksikan harga saham'% (trainScore) )

st.write('jadi didapatkan rmse untuk hasil test sebesar  %.2f, jadi bisa dibilang model masih kurang bagus dalam memprediksikan harga saham'% (testScore) )

st.subheader("Prediction Stock Market: Prophet")

st.image('proph.png')
st.markdown('gambar diambil dari https://facebook.github.io/prophet/docs/quick_start.html#python-api')

st.markdown('''berikut kode untuk menggunakan dan mentrain model prophet: \n
        m = Prophet(daily_seasonality = True) 
        m.fit(clos3)
''')
m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(clos3) # fit the model using all data

st.markdown('''
berikut kode untuk melakukan prediksi pada prophet, saya akan melakukan untuk 30 hari kedepan:\n
        future = m.make_future_dataframe(periods=30)
        prediction = m.predict(future)
''')
future = m.make_future_dataframe(periods=30) #we need to specify the number of days in future
prediction = m.predict(future)

st.markdown('berikut plot untuk prediksi prophet')
m.plot(prediction)
plt.title("Prediction of the bca Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
st.pyplot(plt.gcf())

st.markdown('''
jadi bisa dilihat dalam grafik, bahwa garis biru tersebut nilai yang diprediksi dan area biru disekitar garis adalah selang kepercayaan dari harga prediksinya. 
bedasarkan grafik bisa dilihat bahwa titik titik hitam, yang menunjukan data aslinya, kebanyakan berada di selang, jadi bisa dibilang metode prphet cukup bagus untuk prediksi harga saham.
''')

st.markdown('berikut plot perbandingan prediksi prophet, dengan sebenarnya')

figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()
axes.plot(bca.index, clos3['y'], color = 'red', label = 'Real bca Stock close2')
axes.plot(bca.index, prediction['yhat'][:len(clos3)], color = 'blue', label = 'Predicted bca Stock close2')
#axes.xticks(np.arange(0,394,50))
plt.title('bca Stock close2 Prediction')
plt.xlabel('Time')
plt.ylabel('bca Stock close2')
plt.legend()
plt.savefig('bcapred.png')
st.pyplot(plt.gcf())

st.markdown('''jadi bisa dilihat bahwa untuk prediksi harga langsung, data prediksi cuukup jauh dengan data real, tetapi karena data real terus melewati data 
prediksi maka dapat disimpulkan, prophet sekaligus memprediksikan trend dan juga harga langsung''')


pred2=prediction[:len(clos3)]
rmse = math.sqrt(mean_squared_error(clos3[['y']], pred2[['yhat']]))
st.write('rmse Score: %.2f RMSE' % (rmse))

st.write('jadi didapatkan rmse sebesar  %.2f, jadi bisa dibilang model masih kurang bagus dalam memprediksikan harga saham'% (rmse) )

st.subheader("Evalutation and Discussion")
st.markdown('''
jadi berdasarkan hasil, setiap model memiliki caranya sendiri untuk memprediksikan nilai saham
- arima: memprediksikan tren dari saham yang diprediksikan, jadi cukup bagus untuk prediksi longterm, tapi kurang berguna untuk prediksi jangka panjang
- lstm: memprediksikan harga saham secara langsung, jadi cukup bagus untuk prediksi jangka pendek, 
tetapi mungkin dapat niliai yang jauh dari aslinya pada prediksi jangka panjang
- prophet: memprediksikan harga saham dengan tren dan nilai pas maka prophet lebih fleksibel dalam prediksi, tetapi jika terjadi perubahan tiba tiba
prophet mungkin akan menghasilkan prediksi yang salah. \n
dari semua hasil semua memiliki rmse yang cukup besar hal ini terjadi karena terdapat banyak faktor yang mempengaruhi nilai saham, 
maka cukup susah memprediksikan nilai saham dengan akurasi yang sangat tinggi\n
jadi setiap metode memiliki kegunaanya masing masing, tetapi karena faktor yang mempengaruhi nilai saham, metode metode tidak dapat 
memprediksikan saham dengan sangat tepat.
''')

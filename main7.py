import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import openpyxl

# Streamlit App
st.title("Aplikasi Dashboard Prediksi Data Penerbangan")
st.write("Aplikasi ini dapat melakukan prediksi terhadap jumlah penerbangan, penumpang, dan kargo. Untuk melakukan prediksi diperlukan dataset harian atau mingguan yang berisi Tanggal, Jumlah_penerbangan, Jumlah_penumpang, dan Kargo. Untuk dataset mingguan kolom tanggal disesuiakan dengan tanggal perminggu.")

# Sidebar
prediction_type = st.sidebar.selectbox("Pilih Jenis Prediksi", ["Prediksi Harian", "Prediksi Mingguan"])

if prediction_type == "Prediksi Harian":
    # Load dataset harian
    uploaded_file = st.sidebar.file_uploader("Upload dataset harian", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m/%d/%Y')
        st.write(df.head())

        # Input fitur untuk diprediksi
        selected_feature = st.sidebar.selectbox("Pilih fitur yang ingin diprediksi", df.columns[1:])
        if selected_feature is not None:
            st.write(f"Fitur yang dipilih: {selected_feature}")

            # Input jumlah periode/hari kedepan yang ingin diprediksi
            num_periods = st.sidebar.number_input("Masukkan jumlah periode/hari kedepan yang ingin diprediksi", min_value=1, step=1)

            # Checkbox for displaying evaluation results
            show_evaluation = st.sidebar.checkbox("Tampilkan Akurasi  Nilai Evaluasi Model (RMSE dan MAPE)")

            # Tombol prediksi
            if st.sidebar.button("Prediksi"):
                # Normalisasi data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df[[selected_feature]])

                # Windowing
                window_size = 30  # Jumlah hari dalam satu window
                X = []
                y = []
                for i in range(len(scaled_data) - window_size):
                    X.append(scaled_data[i:i + window_size])
                    y.append(scaled_data[i + window_size])

                X = np.array(X)
                y = np.array(y)

                # Pembagian data train dan test
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # Membangun model SARIMA
                # Uji ADF/stasioneritas dan differencing jika p-value > 0.05
                adf_result = adfuller(y_train[:, 0])
                p_value = adf_result[1]
                apply_differencing = False  # Set this flag based on your requirements

                if p_value > 0.05 and apply_differencing:
                    d = 1
                    y_train = np.diff(y_train, axis=0)
                    y_train = y_train[:-1]  # Menggunakan fungsi slicing untuk menghapus elemen terakhir
                else:
                    d = 0
                best_aic = np.inf
                best_order = None
                best_seasonal_order = None

                for p in range(3):
                    for q in range(3):
                        for P in range(3):
                            for Q in range(3):
                                order = (p,d, q)  # (p, d, q)
                                seasonal_order = (P, d, Q, 7)  # (P, D, Q, S)
                                model = SARIMAX(y_train[:, 0], order=order, seasonal_order=seasonal_order)
                                try:
                                    model_fit = model.fit()
                                    aic = model_fit.aic
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_order = order
                                        best_seasonal_order = seasonal_order
                                except:
                                    continue
                model_sarima = SARIMAX(y_train[:, 0], order=best_order, seasonal_order=best_seasonal_order)
                model_fit_sarima = model_sarima.fit()

                # Membangun model RNN
                model_rnn = Sequential()
                model_rnn.add(LSTM(64, input_shape=(window_size, 1)))
                model_rnn.add(Dense(1))
                model_rnn.compile(optimizer='adam', loss='mse')
                model_rnn.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

                # Prediksi dengan model SARIMA
                sarima_pred = model_fit_sarima.forecast(steps=num_periods)
                sarima_pred = scaler.inverse_transform(sarima_pred.reshape(-1, 1))

                # Prediksi dengan model RNN
                rnn_pred = []
                last_30_days = scaled_data[-window_size:]
                for _ in range(num_periods):
                    input_data = np.reshape(last_30_days, (1, window_size, 1))
                    prediction = model_rnn.predict(input_data)
                    rnn_pred.append(prediction[0])
                    last_30_days = np.append(last_30_days[1:], prediction, axis=0)
                rnn_pred = scaler.inverse_transform(np.array(rnn_pred).reshape(-1, 1))

                # Evaluasi model SARIMA
                y_test_pred_sarima = model_fit_sarima.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
                y_test_pred_sarima = scaler.inverse_transform(y_test_pred_sarima.reshape(-1, 1))
                test_rmse_sarima = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), y_test_pred_sarima))
                test_mape_sarima = mean_absolute_percentage_error(scaler.inverse_transform(y_test),
                                                                   y_test_pred_sarima) * 100

                # Evaluasi model RNN
                y_test_pred_rnn = model_rnn.predict(X_test)
                y_test_pred_rnn = scaler.inverse_transform(y_test_pred_rnn)
                test_rmse_rnn = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), y_test_pred_rnn))
                test_mape_rnn = mean_absolute_percentage_error(scaler.inverse_transform(y_test), y_test_pred_rnn) * 100

                # Tampilkan hasil prediksi dan evaluasi
                st.subheader("Hasil Prediksi untuk Beberapa Hari ke Depan")
                predictions_df = pd.DataFrame({"Tanggal": pd.date_range(start=df['Tanggal'].iloc[-1],
                                                                        periods=num_periods + 1)[1:],
                                                "SARIMA": sarima_pred.flatten(),
                                                "RNN": rnn_pred.flatten()})
                st.write(predictions_df)

                if show_evaluation:  # Show evaluation results if the checkbox is checked
                    st.subheader("Evaluasi Model")
                    evaluations = pd.DataFrame({"Model": ["SARIMA", "RNN"],
                                                "RMSE": [test_rmse_sarima, test_rmse_rnn],
                                                "MAPE": [test_mape_sarima, test_mape_rnn]})
                    st.write(evaluations)

                # Simpan hasil prediksi ke fileCSV
                predictions_df.to_csv("hasil_prediksi.csv", index=False)
                st.download_button(label="Download Hasil Prediksi", data=predictions_df.to_csv(),
                                   file_name="hasil_prediksi.csv")

                # Plot hasil prediksi
                plt.figure(figsize=(12, 4))
                plt.plot(df['Tanggal'][window_size:train_size+window_size], scaler.inverse_transform(y_train).flatten(), label='Train')
                plt.plot(df['Tanggal'][train_size+window_size:], scaler.inverse_transform(y_test).flatten(), label='Test')
                plt.plot(df['Tanggal'][train_size+window_size:], y_test_pred_sarima.flatten(), label='SARIMA Prediction')
                plt.plot(df['Tanggal'][train_size+window_size:], y_test_pred_rnn.flatten(), label='RNN Prediction')
                plt.plot(predictions_df['Tanggal'], predictions_df['SARIMA'], label='SARIMA Future Prediction')
                plt.plot(predictions_df['Tanggal'], predictions_df['RNN'], label='RNN Future Prediction')
                plt.xlabel('Tanggal')
                plt.ylabel(f'Jumlah {selected_feature}')
                plt.title(f'Train-Test Comparison - Jumlah {selected_feature}')
                plt.legend()
                st.pyplot(plt)

elif prediction_type == "Prediksi Mingguan":
    # Load dataset mingguan
    uploaded_file = st.sidebar.file_uploader("Upload dataset mingguan", type="csv")
    if uploaded_file is not None:
        df_weekly = pd.read_csv(uploaded_file)
        df_weekly['Tanggal'] = pd.to_datetime(df_weekly['Tanggal'], format='%m/%d/%Y')
        st.write(df_weekly.head())

        # Input fitur untuk diprediksi
        selected_feature_weekly = st.sidebar.selectbox("Pilih fitur yang ingin diprediksi", df_weekly.columns[1:])
        if selected_feature_weekly is not None:
            st.write(f"Fitur yang dipilih: {selected_feature_weekly}")

            # Input jumlah minggu kedepan yang ingin diprediksi
            num_weeks = st.sidebar.number_input("Masukkan jumlah minggu kedepan yang ingin diprediksi", min_value=1, step=1)

            # Checkbox for displaying evaluation results
            show_evaluation_weekly = st.sidebar.checkbox("Tampilkan Akurasi Nilai Evaluasi Model (RMSE dan MAPE)")

            # Tombol prediksi
            if st.sidebar.button("Prediksi"):
                # Normalisasi data
                scaler_weekly = MinMaxScaler(feature_range=(0, 1))
                scaled_data_weekly = scaler_weekly.fit_transform(df_weekly[[selected_feature_weekly]])

                # Windowing
                window_size_weekly = 12  # Jumlah hari dalam satu window
                X_weekly = []
                y_weekly = []
                for i in range(len(scaled_data_weekly) - window_size_weekly):
                    X_weekly.append(scaled_data_weekly[i:i + window_size_weekly])
                    y_weekly.append(scaled_data_weekly[i + window_size_weekly])

                X_weekly = np.array(X_weekly)
                y_weekly = np.array(y_weekly)

                # Pembagian data train dan test
                train_size_weekly = int(len(X_weekly) * 0.8)
                X_train_weekly, X_test_weekly = X_weekly[:train_size_weekly], X_weekly[train_size_weekly:]
                y_train_weekly, y_test_weekly = y_weekly[:train_size_weekly], y_weekly[train_size_weekly:]

                # Membangun model SARIMA
                # Uji ADF/stasioneritas dan differencing jika p-value > 0.05
                adf_result_weekly = adfuller(y_train_weekly[:, 0])
                p_value_weekly = adf_result_weekly[1]
                apply_differencing_weekly = False  # Set this flag based on your requirements

                if p_value_weekly > 0.05 and apply_differencing_weekly:
                    d_weekly = 1
                    y_train_weekly = np.diff(y_train_weekly, axis=0)
                    y_train_weekly = y_train_weekly[:-1]  # Menggunakan fungsi slicing untuk menghapus elemen terakhir
                else:
                    d_weekly = 0
                best_aic_weekly = np.inf
                best_order_weekly = None
                best_seasonal_order_weekly = None

                for p_weekly in range(3):
                    for q_weekly in range(3):
                        for P_weekly in range(3):
                            for Q_weekly in range(3):
                                order_weekly = (p_weekly, d_weekly, q_weekly)  # (p, d, q)
                                seasonal_order_weekly = (P_weekly, d_weekly, Q_weekly, 12)  # (P, D, Q, S)
                                model_weekly = SARIMAX(y_train_weekly[:, 0], order=order_weekly, seasonal_order=seasonal_order_weekly)
                                try:
                                    model_fit_weekly = model_weekly.fit()
                                    aic_weekly = model_fit_weekly.aic
                                    if aic_weekly < best_aic_weekly:
                                        best_aic_weekly = aic_weekly
                                        best_order_weekly = order_weekly
                                        best_seasonal_order_weekly = seasonal_order_weekly
                                except:
                                    continue
                model_sarima_weekly = SARIMAX(y_train_weekly[:, 0], order=best_order_weekly, seasonal_order=best_seasonal_order_weekly)
                model_fit_sarima_weekly = model_sarima_weekly.fit()

                # Membangun model RNN
                model_rnn_weekly = Sequential()
                model_rnn_weekly.add(LSTM(64, input_shape=(window_size_weekly, 1)))
                model_rnn_weekly.add(Dense(1))
                model_rnn_weekly.compile(optimizer='adam', loss='mse')
                model_rnn_weekly.fit(X_train_weekly, y_train_weekly, epochs=100, batch_size=16, verbose=1)

                # Prediksi dengan model SARIMA
                sarima_pred_weekly = model_fit_sarima_weekly.forecast(steps=num_weeks)
                sarima_pred_weekly = scaler_weekly.inverse_transform(sarima_pred_weekly.reshape(-1, 1))

                # Prediksi dengan model RNN
                rnn_pred_weekly = []
                last_7_days = scaled_data_weekly[-window_size_weekly:]
                for _ in range(num_weeks):
                    input_data_weekly = np.reshape(last_7_days, (1, window_size_weekly, 1))
                    prediction_weekly = model_rnn_weekly.predict(input_data_weekly)
                    rnn_pred_weekly.append(prediction_weekly[0])
                    last_7_days = np.append(last_7_days[1:], prediction_weekly, axis=0)
                rnn_pred_weekly = scaler_weekly.inverse_transform(np.array(rnn_pred_weekly).reshape(-1, 1))

                # Evaluasi model SARIMA
                y_test_pred_sarima_weekly = model_fit_sarima_weekly.predict(start=len(y_train_weekly), end=len(y_train_weekly) + len(y_test_weekly) - 1)
                y_test_pred_sarima_weekly = scaler_weekly.inverse_transform(y_test_pred_sarima_weekly.reshape(-1, 1))
                test_rmse_sarima_weekly = np.sqrt(mean_squared_error(scaler_weekly.inverse_transform(y_test_weekly), y_test_pred_sarima_weekly))
                test_mape_sarima_weekly = mean_absolute_percentage_error(scaler_weekly.inverse_transform(y_test_weekly), y_test_pred_sarima_weekly) * 100

                # Evaluasi model RNN
                y_test_pred_rnn_weekly = model_rnn_weekly.predict(X_test_weekly)
                y_test_pred_rnn_weekly = scaler_weekly.inverse_transform(y_test_pred_rnn_weekly)
                test_rmse_rnn_weekly = np.sqrt(mean_squared_error(scaler_weekly.inverse_transform(y_test_weekly), y_test_pred_rnn_weekly))
                test_mape_rnn_weekly = mean_absolute_percentage_error(scaler_weekly.inverse_transform(y_test_weekly), y_test_pred_rnn_weekly) * 100

                # Tampilkan hasil prediksi dan evaluasi
                st.subheader(f"Hasil Prediksi untuk {num_weeks} Minggu ke Depan")
                predictions_df_weekly = pd.DataFrame({"Tanggal": pd.date_range(start=df_weekly['Tanggal'].iloc[-1],
                                                                            periods=num_weeks + 1)[1:],
                                                    "SARIMA": sarima_pred_weekly.flatten(),
                                                    "RNN": rnn_pred_weekly.flatten()})
                st.write(predictions_df_weekly)

                if show_evaluation_weekly:  # Show evaluation results if the checkbox is checked
                    st.subheader("Evaluasi Model")
                    evaluations_weekly = pd.DataFrame({"Model": ["SARIMA", "RNN"],
                                                        "RMSE": [test_rmse_sarima_weekly, test_rmse_rnn_weekly],
                                                        "MAPE": [test_mape_sarima_weekly, test_mape_rnn_weekly]})
                    st.write(evaluations_weekly)

                # Simpan hasil prediksi ke file CSV
                predictions_df_weekly.to_csv("hasil_prediksi_mingguan.csv", index=False)
                st.download_button(label="Download Hasil Prediksi", data=predictions_df_weekly.to_csv(),
                                   file_name="hasil_prediksi_mingguan.csv")

                # Plot hasil prediksi
                plt.figure(figsize=(12, 4))
                plt.plot(df_weekly['Tanggal'][window_size_weekly:train_size_weekly + window_size_weekly], scaler_weekly.inverse_transform(y_train_weekly).flatten(), label='Train')
                plt.plot(df_weekly['Tanggal'][train_size_weekly + window_size_weekly:], scaler_weekly.inverse_transform(y_test_weekly).flatten(), label='Test')
                plt.plot(df_weekly['Tanggal'][train_size_weekly + window_size_weekly:], y_test_pred_sarima_weekly.flatten(), label='SARIMA Prediction')
                plt.plot(df_weekly['Tanggal'][train_size_weekly + window_size_weekly:], y_test_pred_rnn_weekly.flatten(), label='RNN Prediction')
                plt.plot(predictions_df_weekly['Tanggal'], predictions_df_weekly['SARIMA'], label='SARIMA Future Prediction')
                plt.plot(predictions_df_weekly['Tanggal'], predictions_df_weekly['RNN'], label='RNN Future Prediction')
                plt.xlabel('Tanggal')
                plt.ylabel(f'Jumlah {selected_feature_weekly}')
                plt.title(f'Train-Test Comparison - Jumlah {selected_feature_weekly}')
                plt.legend()
                st.pyplot(plt)


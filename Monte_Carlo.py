import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Laster ned historiske aksjekursdata for Apple
aksjeticker = "AAPL"
start_dato = "2018-09-01"
slutt_dato = "2023-09-01"
apple_data = yf.download(aksjeticker, start=start_dato, end=slutt_dato)

# Beregner daglige avkastninger
avkastninger = np.log(1 + apple_data["Adj Close"].pct_change()).dropna()

# Estimerer parametere (gjennomsnitt og volatilitet) fra historiske avkastninger
mu, sigma = avkastninger.mean(), avkastninger.std()

# Definerer simuleringsparametere
T = 5  # Antall år å simulere
N = 252 * T  # Antall handelsdager på ett år (antatt 252)
simuleringsdager = N

# Antall simuleringer
antall_simuleringer = 1

# Initialiserer en matrise for å lagre resultatene av hver simulering
simulerte_priser_alle = np.zeros((simuleringsdager, antall_simuleringer))

# Kjører n-antall simuleringer
for i in range(antall_simuleringer):
    # Simulerer fremtidige avkastninger ved bruk av normalfordeling
    simulerte_avkastninger = np.random.normal(mu, sigma, simuleringsdager)

    # Beregner simulerte priser
    initial_pris = apple_data["Adj Close"].iloc[-1]
    simulerte_priser = initial_pris * np.exp(np.cumsum(simulerte_avkastninger))
    simulerte_priser_alle[:, i] = simulerte_priser

# Beregner gjennomsnittet av alle simuleringene
gjennomsnittlige_simulerte_priser = np.mean(simulerte_priser_alle, axis=1)

# Oppretter en tidslinje for de simulerte prisene
dato_intervall = pd.date_range(start=slutt_dato, periods=simuleringsdager, freq='B')

# Plotter historiske og gjennomsnittlige simulerte prisdata
plt.figure(figsize=(12, 6))
plt.plot(apple_data.index, apple_data['Adj Close'], label='Historiske Priser')
plt.plot(dato_intervall, gjennomsnittlige_simulerte_priser, label='Gjennomsnittlige Simulerte Priser', color='red')
plt.xlabel('Dato')
plt.ylabel('Aksjekurs')
plt.title('Aksjepris for Apple Inc. og Gjennomsnittlig Simulering for de Neste 5 Årene')
plt.legend()
plt.grid(True)
plt.show()


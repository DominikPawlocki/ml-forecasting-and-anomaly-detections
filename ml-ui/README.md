# Microsoft ML demo

# ML preditions and anomaly detections with UI
This project shows how ML.NET data anomalies detector and forecaster works.
![Generil](/screenshots/Generic.jpg?raw=true)

## Data generators (with discrepancies 'injected'): 
- Random
- Linear
- Sinus

## Anomaly Detections :
- **Spikes** (Transforms.DetectIidSpike() and Transforms.DetectSpikeBySsa() from TimeSeriesCatalog) 
- **Anomalies** (AnomalyDetection.DetectEntireAnomalyBySrCnn from TimeSeriesCatalog) 
- **Changepoints** (not working yet)

## Predictions 
- **Singular Spectrum Analysis (SSA)** based (Forecasting.ForecastBySsa)
![SSAPrediction](/screenshots/SsaPrediction.jpg?raw=true)
- **Regression supervised learning** with several trainers to be choosen 
![Algorithms](/screenshots/RegressionTrainers.jpg?raw=true)




##Remarks

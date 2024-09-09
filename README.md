# Microsoft ML demo

# ML preditions and anomaly detections with UI
This project shows how ML.NET data anomalies detector and forecaster works.
![Generic](/screenshots/Generic.JPG?raw=true)

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
![SSAPrediction](/screenshots/SsaPrediction.JPG?raw=true)
- **Regression supervised learning** with several trainers to be choosen 
![Algorithms](/screenshots/RegressionTrainers.JPG?raw=true)




##Remarks

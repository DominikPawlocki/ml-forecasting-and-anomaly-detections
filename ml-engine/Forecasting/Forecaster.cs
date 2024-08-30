using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using ml_data;

namespace ml_engine.Forecasting
{
    public interface IMlForecaster
    {
        MlForecastResult MlForecast(string detectionByColumnName,
                                    int howManyDataPointsToPredict,
                                    int winSize,
                                    int serLen,
                                    int trnSize,
                                    IEnumerable<DateData> driverData);
    }

    public class Forecaster : IMlForecaster
    {
        public MLContext MlContext { get; }

        public Forecaster()
        {
            MlContext = new MLContext(1);
        }

        public MlForecastResult MlForecast(string detectionByColumnName,
                                           int howManyDataPointsToPredict, int winSize, int serLen, int trnSize,
                                           IEnumerable<DateData> driverData)
        {
            var orderedData = driverData.OrderBy(d => d.Date).ToList();
            //cutTrailingZeroes();

            var dataView = MlContext.Data.LoadFromEnumerable(orderedData);

            MlForecastResult DoPrediction()
            {
                var r = ForecastDriver<DateData>(data: dataView,
                                                                    detectionByColumnName: detectionByColumnName,
                                                                    windowSize: winSize, //-->
                                                                    seriesLength: serLen, //-->
                                                                    trainSize: trnSize, //-->
                                                                    horizon: howManyDataPointsToPredict,
                                                                    isAdaptive: true,
                                                                    confidence: 98 / 1000);
                return r;
            }

            MlForecastResult resultData = DoPrediction();

            return resultData;
        }

        private MlForecastResult ForecastDriver<T>(IDataView data,
                                                   string detectionByColumnName,
                                                   int windowSize,
                                                   int seriesLength,
                                                   int trainSize,
                                                   int horizon,
                                                   bool isAdaptive,
                                                   float confidence) where T : class
        {
            var forecastChain = GetForecastingPipeline(detectionByColumnName, windowSize, seriesLength, trainSize, horizon, isAdaptive, confidence);

            return TrainModel<T>(data, forecastChain);
        }

        private SsaForecastingEstimator GetForecastingPipeline(string detectionByColumnName,
                                                               int windowSize,
                                                               int seriesLength,
                                                               int trainSize,
                                                               int horizon,
                                                               bool isAdaptive,
                                                               float confidence)
        {
            // Instantiate the forecasting model. After https://github.com/dotnet/machinelearning-samples/blob/main/samples/csharp/end-to-end-apps/Forecasting-Sales/README.md
            return MlContext.Forecasting.ForecastBySsa(
                                                       //This is the name of the column that will be used to store predictions. The column must be a vector of type Single.
                                                       outputColumnName: nameof(MlForecastResult.Predictions).ToString(),
                                                       inputColumnName: detectionByColumnName,
                                                       //--> the window for analyzis - eg 7 means a week window.
                                                       //This is the most important parameter that you can use to tune the accuracy of the model for your scenario.
                                                       //Specifically, this parameter is used to define a window of time that is used by the algorithm to decompose the time series data into seasonal/periodic and noise components.
                                                       //Typically, you should start with the largest window size that is representative of the seasonal/periodic business cycle for your scenario.
                                                       //For example, if the business cycle is known to have both weekly and monthly (e.g. 30-day) seasonalities/periods and the data is collected daily,
                                                       //the window size in this case should be 30 to represent the largest window of time that exists in the business cycle. If the same data also exhibits annual seasonality/periods (e.g. 365-day),
                                                       //but the scenario in which the model will be used is not interested in annual seasonality/periods, then the window size does not need to be 365.
                                                       //In this sample, the product data is based on a 12 month cycle where data is collected monthly -- as a result, the window size used is 12.
                                                       windowSize: windowSize,
                                                       //--> the trainSize can be grupped into this 'intervals' eg 365(year) and 30 (month) series
                                                       //This parameter specifies the number of data points that are used when performing a forecast.
                                                       seriesLength: seriesLength,
                                                       //--> overall number of samples to do training - eg 365 as year
                                                       //This parameter specifies the total number of data points in the input time series, starting from the beginning. Note that, after a model is created, it can be saved and updated with new data points that are collected.
                                                       trainSize: trainSize,
                                                       //-->  When determining what the forecasted value for the next period(s) is, the values from previous seven days are used to make a prediction. The model is set to forecast seven periods into the future as defined by the horizon parameter. Because a forecast is an informed guess, it's not always 100% accurate
                                                       //This parameter indicates the number of time periods to predict/forecast. In this sample, we specify 2 to indicate that the next 2 months of product units will be predicated/forecasted.
                                                       horizon: horizon,
                                                       isAdaptive: isAdaptive,
                                                       discountFactor: 1f,
                                                       rankSelectionMethod: RankSelectionMethod.Exact,
                                                       rank: null,
                                                       maxRank: null,
                                                       shouldStabilize: true,
                                                       shouldMaintainInfo: false,
                                                       maxGrowth: null,
                                                       //This is the name of the column that will be used to store the lower confidence interval bound for each forecasted value. The ProductUnitTimeSeriesPrediction class also contains this output column.
                                                       confidenceLowerBoundColumn: nameof(MlForecastResult.ConfidenceLowerBounds).ToString(),
                                                       //This is the name of the column that will be used to store the upper confidence interval bound for each forecasted value. The ProductUnitTimeSeriesPrediction class also contains this output column.
                                                       confidenceUpperBoundColumn: nameof(MlForecastResult.ConfidenceUpperBounds).ToString(),
                                                       //This parameter indicates the likelihood the real observed value will fall within the specified interval bounds. Typically, .95 is an acceptable starting point - this value should be between [0, 1). Usually, the higher the confidence level, the wider the range that the interval bounds will be. And conversely, the lower the confidence level, the narrower the interval bounds.
                                                       confidenceLevel: confidence,
                                                       variableHorizon: false);
        }

        private MlForecastResult TrainModel<T>(IDataView dataFor,
                                               SsaForecastingEstimator forecastingModelChain,
                                               bool ignoreMissingColumns = true) where T : class
        {
            // Train.
            var transformer = forecastingModelChain.Fit(dataFor);

            // Forecast next X (same like series amount) values.
            var forecastEngine = transformer.CreateTimeSeriesEngine<T, MlForecastResult>(MlContext, ignoreMissingColumns);

            var forecast = forecastEngine.Predict();
            return forecast;
        }
    }
}

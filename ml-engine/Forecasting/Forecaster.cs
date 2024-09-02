using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.TimeSeries;
using ml_data;
using System.Globalization;

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

        IEnumerable<MlLinearRegressionDateValuePredition> ForecastByLinearRegression(TransformerChain<ITransformer> trainedModel,
                                                                                     IEnumerable<DateData> dataPointsToBePredictedByModel,
                                                                                     bool ignoreMissingColumns = true);

        (IEnumerable<MlLinearRegressionDateValuePredition> trainedModelDataOutput, TransformerChain<ITransformer> trainedModel)
            TrainModelAndReturnLearntOutput(string regressionLearnerName, string detectionByColumnName, IEnumerable<DateData> allDataPointsUsedFortraining);

    }

    public class Forecaster : IMlForecaster
    {
        public MLContext MlContext { get; }

        public Forecaster()
        {
            MlContext = new MLContext(0);
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
                var r = ForecastBySsa<DateData>(data: dataView,
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

        public IEnumerable<MlLinearRegressionDateValuePredition> ForecastByLinearRegression(TransformerChain<ITransformer> trainedModel,
                                                                                            IEnumerable<DateData> dataPointsToBePredictedByModel,
                                                                                            bool ignoreMissingColumns = true)
        {
            // Create prediction engine related to the loaded trained model
            var predEngine = MlContext.Model.CreatePredictionEngine<DateData, MlLinearRegressionDateValuePredition>(trainedModel);
            var result = new List<MlLinearRegressionDateValuePredition>(dataPointsToBePredictedByModel.Count());
            foreach (var dataPoint in dataPointsToBePredictedByModel)
            {
                var forecast = predEngine.Predict(new DateData(dataPoint.YearForMl, dataPoint.MonthForMl, dataPoint.DayForMl)); // ,0 -> this value is to be predicted for given date
                result.Add(forecast);
            }
            return result;
        }

        public (IEnumerable<MlLinearRegressionDateValuePredition> trainedModelDataOutput, TransformerChain<ITransformer> trainedModel)
            TrainModelAndReturnLearntOutput(string regressionLearnerName,
                                            string detectionByColumnName,
                                            //IEnumerable<DateData> dataPointsToBePredictedByModel,
                                            IEnumerable<DateData> driverData)
        {
            var orderedData = driverData.OrderBy(d => d.Date).ToList();

            var dataView = MlContext.Data.LoadFromEnumerable(orderedData);

            var r = ForecastByRegression(regressionLearnerName, dataView, detectionByColumnName, driverData);
            return r;
        }

        private MlForecastResult ForecastBySsa<T>(IDataView data,
                                                  string detectionByColumnName,
                                                  int windowSize,
                                                  int seriesLength,
                                                  int trainSize,
                                                  int horizon,
                                                  bool isAdaptive,
                                                  float confidence) where T : class
        {
            var forecastChain = GetForecastingSsaPipeline(detectionByColumnName, windowSize, seriesLength, trainSize, horizon, isAdaptive, confidence);

            return PredictBySsa<T>(data, forecastChain);
        }

        private (IEnumerable<MlLinearRegressionDateValuePredition> trainedModelDataOutput, TransformerChain<ITransformer> trainedModel)
            ForecastByRegression(string regressionLearnerName, IDataView data,//IEnumerable<DateData> dataPointsToBePredictedByModel,
                                                                                       string detectionByColumnName,
                                                                                       IEnumerable<DateData> driverData)
        {
            // STEP 2: Common data process configuration with pipeline data transformations
            //var dataProcessPipeline = MlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(TaxiTrip.FareAmount))
            //                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(TaxiTrip.VendorId)))
            //                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: nameof(TaxiTrip.RateCode)))
            //                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(TaxiTrip.PaymentType)))
            //                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.PassengerCount)))
            //                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.TripTime)))
            //                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.TripDistance)))
            //                            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", nameof(TaxiTrip.PassengerCount)
            //                            , nameof(TaxiTrip.TripTime), nameof(TaxiTrip.TripDistance)));

            Action<System.String, DateTime> mapping = (input, output) =>
            {
                const string DATETIME_FORMAT = "dd/MM/yyyy";
                //output.Col1 = input.Col1;

                if (DateTime.TryParseExact(input,
                                            DATETIME_FORMAT,
                                            CultureInfo.InvariantCulture,
                                            DateTimeStyles.None, out var result))
                    output = result;
            };


            // It needs am additional column usually called 'Feature' 
            Func<string, string> columnNameToSingleTransformer = (string colName) => ($"{colName}Single");

            //var dataProcessPipeline = MlContext.Transforms.CopyColumns(outputColumnName: detectionByColumnName, inputColumnName: detectionByColumnName)

            //                            //.Append(MlContext.Transforms.Conversion.ConvertType(new[] {
            //                            //    new InputOutputColumnPair("DateEncoded", inputColumnName: nameof(DateData.Date))}, DataKind.Single))
            //                            .Append(MlContext.Transforms.CopyColumns(nameof(DateData.YearForMl), nameof(DateData.YearForMl)))
            //                            .Append(MlContext.Transforms.CopyColumns(nameof(DateData.MonthForMl), nameof(DateData.MonthForMl)))
            //                            .Append(MlContext.Transforms.CopyColumns(nameof(DateData.DayForMl), nameof(DateData.DayForMl)))


            //                            //.Append(MlContext.Transforms.CopyColumns(outputColumnName: "DateEncoded", inputColumnName: nameof(Date.Date)))
            //                            //.Append(MlContext.Transforms.NormalizeMeanVariance(outputColumnName: "DateEncodedV", "DateEncoded"))
            //                            //.Append(MlContext.Transforms.Conversion.ConvertType(new[] {
            //                            //    new InputOutputColumnPair(columnNameToSingleTransformer(nameof(DateData.Value)), nameof(DateData.Value))}, DataKind.Single))
            //                            .Append(MlContext.Transforms.Concatenate("Features", nameof(DateData.YearForMl), nameof(DateData.MonthForMl), nameof(DateData.DayForMl)))
            //                            .AppendCacheCheckpoint(MlContext); // Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            var dataProcessPipeline = MlContext.Transforms.
                Concatenate("Features", nameof(DateData.YearForMl), nameof(DateData.MonthForMl), nameof(DateData.DayForMl))
                            .AppendCacheCheckpoint(MlContext); // Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            var (name, trainerChoosen) = FetchAllRegressionLearners(labelColumnName: detectionByColumnName, featureColumnName: "Features").FirstOrDefault(r => r.name == regressionLearnerName);

            var trainingPipeline = dataProcessPipeline.Append(trainerChoosen);

            // Train.
            var trainedModel = trainingPipeline.Fit(data);

            //-------------- model evaluation -------------------
            var predictions = trainedModel.Transform(data);
            //var metrics = MlContext.Regression.Evaluate(predictions, nameof(DateData.ValueForMl));
            //-------------------------------------------------
            //------------ just output all predictions 'learnt' - to see what model output gives with comparison to original data
            var predEngine = MlContext.Model.CreatePredictionEngine<DateData, MlLinearRegressionDateValuePredition>(trainedModel);
            var result = new List<MlLinearRegressionDateValuePredition>(driverData.Count());
            foreach (var dataPoint in driverData)
            {
                var forecast = predEngine.Predict(new DateData(dataPoint.YearForMl, dataPoint.MonthForMl, dataPoint.DayForMl)); // ,0 -> this value is to be predicted for given date
                result.Add(forecast);
            }

            return (result, trainedModel); //in memory, be cautious when doing like with with bigger dataSets, rather use Save and Load methods via file or Stream
        }

        private SsaForecastingEstimator GetForecastingSsaPipeline(string detectionByColumnName,
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

        private SdcaRegressionTrainer GetForecastingRegressionPipeline(string detectionByColumnName,
                                                                       string featureColumnName,
                                                                       ISupportSdcaRegressionLoss? lossFunction = null,
                                                                       float? l2Regularization = null,
                                                                       float? l1Regularization = null,
                                                                       int? maximumNumberOfIterations = null)
        {
            return MlContext.Regression.Trainers.Sdca(labelColumnName: detectionByColumnName,
                                                      featureColumnName: featureColumnName,
                                                      null,
                                                      lossFunction,
                                                      l2Regularization,
                                                      l2Regularization,
                                                      maximumNumberOfIterations);

        }

        private MlForecastResult PredictBySsa<T>(IDataView dataFor,
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

        private (string name, IEstimator<ITransformer> value)[] FetchAllRegressionLearners(string labelColumnName, string featureColumnName)
        {
            return [
            ("FastTree", MlContext.Regression.Trainers.FastTree(labelColumnName, featureColumnName)),
            ("Poisson", MlContext.Regression.Trainers.LbfgsPoissonRegression( labelColumnName, featureColumnName)),
                ("SDCA", MlContext.Regression.Trainers.Sdca( labelColumnName, featureColumnName)),
                ("FastTreeTweedie", MlContext.Regression.Trainers.FastTreeTweedie(labelColumnName,featureColumnName)),
                //Other possible learners that could be included
                //...FastForestRegressor...
                //...GeneralizedAdditiveModelRegressor...
                //...OnlineGradientDescent... (Might need to normalize the features first)
            ];

        }
    }
}
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.TimeSeries;
using ml_data;
using System.Data;

namespace ml_engine.Forecasting
{
    public interface IMlForecaster
    {
        (IEnumerable<MlSSAPrediction> trainedModelDataOutput, TransformerChain<SsaForecastingTransformer> trainedModel)
            SSATrainModelAndReturnLearntOutput(string detectionByColumnName,
                                               int winSize,
                                               int serLen,
                                               int trnSize,
                                               bool isAdaptive,
                                               int confidence,
                                               IEnumerable<DateData> driverData);
        MlSSAPrediction ForecastBySSA(TransformerChain<SsaForecastingTransformer> trainedModel,
                                       DateTime predictionsStartingDate,
                                       int howManyDataPointsToPredict,
                                       bool ignoreMissingColumns = true);

        IEnumerable<MlLinearRegressionDateValuePrediction> ForecastByLinearRegression(TransformerChain<ITransformer> trainedModel,
                                                                                     IEnumerable<DateData> dataPointsToBePredictedByModel,
                                                                                     bool ignoreMissingColumns = true);

        (IEnumerable<MlLinearRegressionDateValuePrediction> trainedModelDataOutput, TransformerChain<ITransformer> trainedModel)
            TrainModelAndReturnLearntOutput(string regressionLearnerName, string detectionByColumnName, IEnumerable<DateData> allDataPointsUsedFortraining);

    }

    public class Forecaster : IMlForecaster
    {
        public MLContext MlContext { get; }

        public Forecaster()
        {
            MlContext = new MLContext(0);
        }

        public (IEnumerable<MlSSAPrediction> trainedModelDataOutput, TransformerChain<SsaForecastingTransformer> trainedModel)
            SSATrainModelAndReturnLearntOutput(string detectionByColumnName,
                                               int winSize,
                                               int serLen,
                                               int trnSize,
                                               bool isAdaptive,
                                               int confidence,
                                               IEnumerable<DateData> driverData)

        {
            var orderedData = driverData.OrderBy(d => d.Date).ToList();
            var dataView = MlContext.Data.LoadFromEnumerable(orderedData);

            var r = TrainSSAModel<DateData>(data: dataView,
                                            driverData: driverData,
                                            detectionByColumnName: detectionByColumnName,
                                            windowSize: winSize, //-->
                                            seriesLength: serLen, //-->
                                            trainSize: trnSize, //-->
                                            horizon: 1,
                                            isAdaptive: isAdaptive,
                                            confidence: confidence / 100);
            return r;
        }

        public MlSSAPrediction ForecastBySSA(TransformerChain<SsaForecastingTransformer> trainedModel,
                                              DateTime predictionsStartingDate,
                                              int howManyDataPointsToPredict,
                                              bool ignoreMissingColumns = true)
        {
            // Create prediction engine related to the loaded trained model
            var forecastEngine = trainedModel.CreateTimeSeriesEngine<DateData, MlSSAPrediction>(MlContext, ignoreMissingColumns);

            return forecastEngine.Predict(howManyDataPointsToPredict);
        }

        public IEnumerable<MlLinearRegressionDateValuePrediction> ForecastByLinearRegression(TransformerChain<ITransformer> trainedModel,
                                                                                            IEnumerable<DateData> dataPointsToBePredictedByModel,
                                                                                            bool ignoreMissingColumns = true)
        {
            // Create prediction engine related to the loaded trained model
            var predEngine = MlContext.Model.CreatePredictionEngine<DateData, MlLinearRegressionDateValuePrediction>(trainedModel);
            var result = new List<MlLinearRegressionDateValuePrediction>(dataPointsToBePredictedByModel.Count());
            foreach (var dataPoint in dataPointsToBePredictedByModel)
            {
                var forecast = predEngine.Predict(dataPoint);
                result.Add(forecast);
            }
            return result;
        }

        public (IEnumerable<MlLinearRegressionDateValuePrediction> trainedModelDataOutput, TransformerChain<ITransformer> trainedModel)
            TrainModelAndReturnLearntOutput(string regressionLearnerName,
                                            string detectionByColumnName,
                                            IEnumerable<DateData> driverData)
        {
            var orderedData = driverData.OrderBy(d => d.Date).ToList();

            var dataView = MlContext.Data.LoadFromEnumerable(orderedData);

            var r = ForecastByRegression(regressionLearnerName, dataView, detectionByColumnName, driverData);
            return r;
        }

        private (IEnumerable<MlSSAPrediction> trainedModelDataOutput, TransformerChain<SsaForecastingTransformer> trainedModel)
            TrainSSAModel<T>(IDataView data,
                             IEnumerable<T> driverData,
                             string detectionByColumnName,
                             int windowSize,
                             int seriesLength,
                             int trainSize,
                             int horizon,
                             bool isAdaptive,
                             float confidence) where T : class
        {
            var forecastChain = GetForecastingSsaPipeline(detectionByColumnName + "Single", windowSize, seriesLength, trainSize, horizon, isAdaptive, confidence);

            var estimatorChain = MlContext.Transforms.Conversion
                .ConvertType(new[] { new InputOutputColumnPair(detectionByColumnName + "Single", detectionByColumnName) }, DataKind.Single)
                    .Append(MlContext.Transforms.Concatenate("Features", detectionByColumnName + "Single")) //needs vector of Single
                    .AppendCacheCheckpoint(MlContext)
                .Append(estimator: forecastChain);

            var trainedModel = estimatorChain.Fit(data);

            //var predictions = trainedModel.Transform(dataFor);
            var ignoreMissingColumns = true;
            var forecastEngine = trainedModel.CreateTimeSeriesEngine<T, MlSSAPrediction>(MlContext, ignoreMissingColumns);

            //------------ just output all predictions 'learnt' - to see what model output gives with comparison to original data
            var result = new List<MlSSAPrediction>(driverData.Count());
            foreach (var dataPoint in driverData)
            {
                result.Add(forecastEngine.Predict(dataPoint));
            }

            return (result, trainedModel); //in memory, be cautious when doing like with with bigger dataSets, rather use Save and Load methods via file or Stream
        }

        private (IEnumerable<MlLinearRegressionDateValuePrediction> trainedModelDataOutput, TransformerChain<ITransformer> trainedModel)
            ForecastByRegression(string regressionLearnerName, IDataView data, string detectionByColumnName, IEnumerable<DateData> driverData)
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

            //Func<string, string> columnNameToSingleTransformer = (string colName) => ($"{colName}Single");

            //var dataProcessPipeline = MlContext.Transforms.CopyColumns(outputColumnName: detectionByColumnName, inputColumnName: detectionByColumnName)
            //                            //.Append(MlContext.Transforms.Conversion.ConvertType(new[] {
            //                            //    new InputOutputColumnPair("DateEncoded", inputColumnName: nameof(DateData.Date))}, DataKind.Single))

            //                            //.Append(MlContext.Transforms.CopyColumns(outputColumnName: "DateEncoded", inputColumnName: nameof(Date.Date)))
            //                            //.Append(MlContext.Transforms.NormalizeMeanVariance(outputColumnName: "DateEncodedV", "DateEncoded"))
            //                            //.Append(MlContext.Transforms.Conversion.ConvertType(new[] {
            //                            //    new InputOutputColumnPair(columnNameToSingleTransformer(nameof(DateData.Value)), nameof(DateData.Value))}, DataKind.Single))
            //                            .Append(MlContext.Transforms.Concatenate("Features", nameof(DateData.YearForMl), nameof(DateData.MonthForMl), nameof(DateData.DayForMl)))
            //                            .AppendCacheCheckpoint(MlContext); // Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            //var dataProcessPipeline = MlContext.Transforms.
            //    NormalizeMeanVariance(outputColumnName: "nYearForMl", nameof(DateData.YearForMl))
            //    .Append(MlContext.Transforms.NormalizeMeanVariance(outputColumnName: "nMonthForMl", nameof(DateData.MonthForMl)))
            //    .Append(MlContext.Transforms.NormalizeMeanVariance(outputColumnName: "nDayForMl", nameof(DateData.DayForMl)))
            //    .Append(MlContext.Transforms.Concatenate("Features", "nYearForMl", "nMonthForMl", "nDayForMl"))
            //                .AppendCacheCheckpoint(MlContext); // Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            //        var dataProcessPipeline = MlContext.Transforms.
            //Concatenate("Features", nameof(DateData.YearForMl), nameof(DateData.MonthForMl), nameof(DateData.DayForMl))
            //.AppendCacheCheckpoint(MlContext);

            //------------ works good in general ---------
            //var dataProcessPipeline = MlContext.Transforms.Conversion
            //    .ConvertType(new[] { new InputOutputColumnPair(detectionByColumnName + "Single", detectionByColumnName) }, DataKind.Single)
            //    .Append(MlContext.Transforms.Conversion.ConvertType(new[] { new InputOutputColumnPair("DateEncoded", inputColumnName: nameof(DateData.Date)) }, DataKind.Single))
            //    .Append(MlContext.Transforms.Concatenate("Features", "DateEncoded")) //needs vector of Single
            //    .AppendCacheCheckpoint(MlContext);
            //-----------------

            var dataProcessPipeline = MlContext.Transforms.Conversion
                .ConvertType(new[] { new InputOutputColumnPair(detectionByColumnName + "Single", detectionByColumnName) }, DataKind.Single)
                .Append(MlContext.Transforms.Concatenate("Features", nameof(DateData.DateProjectedToNumberForMl))) //needs vector of Single
                .AppendCacheCheckpoint(MlContext);

            //------ was good for SDCA --------------
            //var dataProcessPipeline = MlContext.Transforms.Conversion
            //    .ConvertType(new[] { new InputOutputColumnPair(detectionByColumnName + "Single", detectionByColumnName) }, DataKind.Single)
            //    .Append(MlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Features", inputColumnName: nameof(DateData.Date)))
            //    .AppendCacheCheckpoint(MlContext);
            //------------------

            var (name, trainerChoosen) = FetchAllRegressionLearners(labelColumnName: detectionByColumnName + "Single", featureColumnName: "Features").FirstOrDefault(r => r.name == regressionLearnerName);

            var trainingPipeline = dataProcessPipeline.Append(trainerChoosen);

            // Train.
            var trainedModel = trainingPipeline.Fit(data);

            //var predictions = trainedModel.Transform(data);
            //-------------- model evaluation - not used now, shows metrics and has a possibility to train model better-------------------
            //var metrics = MlContext.Regression.Evaluate(predictions, nameof(DateData.ValueForMl));
            //-------------------------------------------------
            //------------ just output all predictions 'learnt' - to see what model output gives with comparison to original data
            var predEngine = MlContext.Model.CreatePredictionEngine<DateData, MlLinearRegressionDateValuePrediction>(trainedModel);

            var result = new List<MlLinearRegressionDateValuePrediction>(driverData.Count());
            foreach (var dataPoint in driverData)
            {
                var forecast = predEngine.Predict(dataPoint);
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
                                                       outputColumnName: nameof(MlSSAPrediction.Predictions).ToString(),
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
                                                       confidenceLowerBoundColumn: nameof(MlSSAPrediction.ConfidenceLowerBounds).ToString(),
                                                       //This is the name of the column that will be used to store the upper confidence interval bound for each forecasted value. The ProductUnitTimeSeriesPrediction class also contains this output column.
                                                       confidenceUpperBoundColumn: nameof(MlSSAPrediction.ConfidenceUpperBounds).ToString(),
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

        private (string name, IEstimator<ITransformer> value)[] FetchAllRegressionLearners(string labelColumnName, string featureColumnName)
        {
            return [
            ("FastTree", MlContext.Regression.Trainers.FastTree(labelColumnName, featureColumnName)),
            ("Poisson", MlContext.Regression.Trainers.LbfgsPoissonRegression( labelColumnName, featureColumnName)),
                ("SDCA", MlContext.Regression.Trainers.Sdca( labelColumnName, featureColumnName)),
                ("FastTreeTweedie", MlContext.Regression.Trainers.FastTreeTweedie(labelColumnName,featureColumnName)),
                ("GBM", MlContext.Regression.Trainers.LightGbm(labelColumnName,featureColumnName)),
                ("OLS", MlContext.Regression.Trainers.Ols(labelColumnName,featureColumnName)),
                ("ODG", MlContext.Regression.Trainers.OnlineGradientDescent(labelColumnName,featureColumnName)),
                ("GAM", MlContext.Regression.Trainers.Gam(labelColumnName,featureColumnName)),
                //("RandomForests", MlContext.Regression.Trainers.fore(labelColumnName,featureColumnName)),
            ];
        }
    }
}
using AutoMapper;
using Microsoft.ML.Data;
using Microsoft.ML;
using ml_data;
using ml_engine.Forecasting;
using ml_ui.ViewModels;
using Microsoft.ML.Transforms.TimeSeries;

namespace ml_ui.Services
{
    public interface IMlForecastingService
    {
        Task<(IEnumerable<DateIntegerForecasterDataViewModel> trainedModelDataOutput, TransformerChain<SsaForecastingTransformer>? trainedModel)> TrainSSA(string detectionByColumnName,
                                                                                                                                                          int howManyDataPointsToPredict,
                                                                                                                                                          int winSize,
                                                                                                                                                          int serLen,
                                                                                                                                                          int trnSize,
                                                                                                                                                          IEnumerable<DateIntegerDataViewModel> dataSet);
        Task<IEnumerable<DateIntegerForecasterDataViewModel>> ForecastBySSA(TransformerChain<SsaForecastingTransformer> trainedSSAModel,
                                                                            DateTime pointsToBePredictedStartDate,
                                                                            int howManyDataPointsToPredict);

        Task<(IEnumerable<DateIntegerForecasterDataViewModel> trainedModelDataOutput, TransformerChain<ITransformer>? trainedModel)> TrainLinearRegression(string regressionLearnerName,
                                                                                                                                                          string detectionByColumnName,
                                                                                                                                                          IEnumerable<DateIntegerDataViewModel> dataSet);
        Task<IEnumerable<DateIntegerForecasterDataViewModel>> ForerecastByLinearRegression(TransformerChain<ITransformer> trainedRegressionModel,
                                                                                           DateTime pointsToBePredictedStartDate,
                                                                                           int howManyDataPointsToPredict);
    }

    public class MlForecastingService : IMlForecastingService
    {
        private readonly IMlForecaster _forecaster;
        private readonly IMapper _mapper;

        public MlForecastingService(IMlForecaster forecaster, IMapper mapper)
        {
            _forecaster = forecaster;
            _mapper = mapper;
        }

        public async Task<(IEnumerable<DateIntegerForecasterDataViewModel> trainedModelDataOutput, TransformerChain<SsaForecastingTransformer>? trainedModel)>
            TrainSSA(string detectionByColumnName,
                     int howManyDataPointsToPredict,
                     int winSize,
                     int serLen,
                     int trnSize,
                     IEnumerable<DateIntegerDataViewModel> dataSet)
        {
            var result = new List<DateIntegerForecasterDataViewModel>();

            if (dataSet == null || !dataSet.Any())
            {
                return (result, null);
            }
            return await Task.Run(() =>
            {
                var dataSetForMl = _mapper.Map<IEnumerable<DateData>>(dataSet).ToList();

                var (trainedModelDataOutput, trainedModel) = _forecaster.SSATrainModelAndReturnLearntOutput(detectionByColumnName,
                                                                                            winSize,
                                                                                            serLen,
                                                                                            trnSize, dataSetForMl);

                var resultsCasted = trainedModelDataOutput.ToArray();
                //No automapper here 

                //no automapper here - this model doesnt work like this. As we train model with making 1 prediction per dataPoint, there is one value in Vector preditions
                for (var i = 0; i < dataSetForMl.Count; i++)
                {
                    result.Add(new DateIntegerForecasterDataViewModel()
                    {
                        Date = dataSetForMl[i].Date,
                        Value = (int)resultsCasted[i].Predictions[0],
                        IsForecasted = true,
                        ConfidenceLowerBound = resultsCasted[i].ConfidenceLowerBounds[0],
                        ConfidenceUpperBound = resultsCasted[i].ConfidenceUpperBounds[0],
                    });
                }
                return (result, trainedModel);
            });
        }

        public async Task<IEnumerable<DateIntegerForecasterDataViewModel>> ForecastBySSA(TransformerChain<SsaForecastingTransformer> trainedSSAModel,
                                                                                                        DateTime pointsToBePredictedStartDate,
                                                                                                        int howManyDataPointsToPredict)
        {
            return await Task.Run(() =>
            {
                if (howManyDataPointsToPredict == 0)
                    return Enumerable.Empty<DateIntegerForecasterDataViewModel>();

                var forecasted = _forecaster.ForecastBySSA(trainedSSAModel, pointsToBePredictedStartDate, howManyDataPointsToPredict);
                var result = new List<DateIntegerForecasterDataViewModel>();
                for (var i = 0; i < howManyDataPointsToPredict; i++)
                {
                    result.Add(new DateIntegerForecasterDataViewModel()
                    {
                        Date = pointsToBePredictedStartDate.AddDays(7 * i),
                        Value = (int)forecasted.Predictions[i],
                        IsForecasted = true,
                        ConfidenceLowerBound = forecasted.ConfidenceLowerBounds[i],
                        ConfidenceUpperBound = forecasted.ConfidenceUpperBounds[i],
                    });
                }
                return result;
            });
        }

        public async Task<IEnumerable<DateIntegerForecasterDataViewModel>> ForerecastByLinearRegression(TransformerChain<ITransformer> trainedRegressionModel,
                                                                                                        DateTime pointsToBePredictedStartDate,
                                                                                                        int howManyDataPointsToPredict)
        {
            return await Task.Run(() =>
            {
                if (howManyDataPointsToPredict == 0)
                    return Enumerable.Empty<DateIntegerForecasterDataViewModel>();

                var toBeRunAgainstModel = new List<DateData>();
                for (var i = 0; i < howManyDataPointsToPredict; i++)
                {
                    pointsToBePredictedStartDate = pointsToBePredictedStartDate.AddDays(i * 7); //we have a weekly data, however it doesnt have to be like this, we can predict any date in the model
                    toBeRunAgainstModel.Add(new DateData(pointsToBePredictedStartDate, 0)); //this 0 is to be predicted (run against trained model)
                }

                var forecast = _forecaster.ForecastByLinearRegression(trainedRegressionModel, toBeRunAgainstModel);
                var result = _mapper.Map<IEnumerable<DateIntegerForecasterDataViewModel>>(forecast);

                return result;
            });
        }

        public async Task<(IEnumerable<DateIntegerForecasterDataViewModel> trainedModelDataOutput, TransformerChain<ITransformer>? trainedModel)>
            TrainLinearRegression(
                string regressionLearnerName,
                string detectionByColumnName,
                IEnumerable<DateIntegerDataViewModel> dataSet)
        {

            if (dataSet == null || !dataSet.Any())
            {
                return ([], null);
            }
            return await Task.Run(() =>
            {
                var dataSetForMl = _mapper.Map<IEnumerable<DateData>>(dataSet);

                var modelOutput = _forecaster.TrainModelAndReturnLearntOutput(regressionLearnerName, detectionByColumnName, dataSetForMl);
                var result = _mapper.Map<IEnumerable<DateIntegerForecasterDataViewModel>>(modelOutput.trainedModelDataOutput);

                return (result, modelOutput.trainedModel);
            });
        }
    }
}

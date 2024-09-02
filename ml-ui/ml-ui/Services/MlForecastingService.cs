using AutoMapper;
using Microsoft.ML.Data;
using Microsoft.ML;
using ml_data;
using ml_engine.Forecasting;
using ml_ui.ViewModels;

namespace ml_ui.Services
{
    public interface IMlForecastingService
    {
        Task<IEnumerable<DateIntegerForecasterDataViewModel>> Forecast(string detectionByColumnName,
                                                                       int howManyDataPointsToPredict,
                                                                       int winSize,
                                                                       int serLen,
                                                                       int trnSize,
                                                                       IEnumerable<DateIntegerDataViewModel> dataSet);
        Task<(IEnumerable<DateIntegerForecasterDataViewModel> trainedModelDataOutput, TransformerChain<ITransformer> trainedModel)> TrainLinearRegression(string regressionLearnerName,
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

        public async Task<IEnumerable<DateIntegerForecasterDataViewModel>> Forecast(string detectionByColumnName,
                                                                                    int howManyDataPointsToPredict,
                                                                                    int winSize,
                                                                                    int serLen,
                                                                                    int trnSize,
                                                                                    IEnumerable<DateIntegerDataViewModel> dataSet)
        {
            if (dataSet == null || !dataSet.Any())
            {
                return [];
            }
            return await Task.Run(() =>
            {
                var dataSetForMl = _mapper.Map<IEnumerable<DateData>>(dataSet);
                if (dataSetForMl is null)
                    return Enumerable.Empty<DateIntegerForecasterDataViewModel>();
                var forecast = _forecaster.MlForecast(detectionByColumnName, howManyDataPointsToPredict, winSize, serLen, trnSize, dataSetForMl);
                var result = new List<DateIntegerForecasterDataViewModel>(forecast.Predictions.Length);
                for (var i = 0; i < forecast.Predictions.Count(); i++)
                {
                    result.Add(new DateIntegerForecasterDataViewModel()
                    {
                        Date = dataSetForMl.OrderBy(d => d.Date).LastOrDefault().Date.AddDays(7 * (i + 1)),
                        Value = (int)forecast.Predictions[i],
                        IsForecasted = true,
                        ConfidenceLowerBound = forecast.ConfidenceLowerBounds[i],
                        ConfidenceUpperBound = forecast.ConfidenceUpperBounds[i],
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
                //var result = new List<DateIntegerForecasterDataViewModel>(howManyDataPointsToPredict);
                var result = _mapper.Map<IEnumerable<DateIntegerForecasterDataViewModel>>(forecast);

                return result;
            });
        }

        public async Task<(IEnumerable<DateIntegerForecasterDataViewModel> trainedModelDataOutput, TransformerChain<ITransformer> trainedModel)>
            TrainLinearRegression(
                string regressionLearnerName,
                string detectionByColumnName,
                IEnumerable<DateIntegerDataViewModel> dataSet)
        {

            if (dataSet == null || !dataSet.Any())
            {
                return (Enumerable.Empty<DateIntegerForecasterDataViewModel>(), null);
            }
            return await Task.Run(() =>
            {
                var dataSetForMl = _mapper.Map<IEnumerable<DateData>>(dataSet);
                if (dataSetForMl is null)
                    return (Enumerable.Empty<DateIntegerForecasterDataViewModel>(), null);

                var modelOutput = _forecaster.TrainModelAndReturnLearntOutput(regressionLearnerName, detectionByColumnName, dataSetForMl);
                var result = _mapper.Map<IEnumerable<DateIntegerForecasterDataViewModel>>(modelOutput.trainedModelDataOutput);

                return (result, modelOutput.trainedModel);
            });
        }
    }
}

using AutoMapper;
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
    }
}

using Microsoft.ML.Transforms.TimeSeries;
using ml_engine.AnomalyDetections;
using ml_data;
using ml_ui.ViewModels;
using System.Data;
using AutoMapper;
using System.Drawing;
using Microsoft.ML;
using static System.Runtime.InteropServices.JavaScript.JSType;
using Newtonsoft.Json.Linq;

namespace ml_ui.Services
{
    interface IMlDataAnomaliesDetectingService
    {
        public Task<IEnumerable<SpikeDetectionDataViewModel>> DetectSpikes(DetectionMethod numericMethod,
                                                                           int pvalueHistoryLength,
                                                                           int ssaTrainingWindowSize,
                                                                           int ssaSeasonalityWindowSize,
                                                                           int confidence,
                                                                           IEnumerable<DateIntegerDataViewModel> dataSet,
                                                                           string detectionByColumnName,
                                                                           bool detectedAlertsOnly = true);

    }
    public class MlDataAnomaliesDetectingService : IMlDataAnomaliesDetectingService
    {
        private readonly ISpikesDetector _spikesDetector;
        private readonly IMapper _mapper;

        public MlDataAnomaliesDetectingService(ISpikesDetector spikesDetector, IMapper mapper)
        {
            _spikesDetector = spikesDetector;
            _mapper = mapper;
        }

        public async Task<IEnumerable<SpikeDetectionDataViewModel>> DetectSpikes(DetectionMethod numericMethod,
                                                                                 int pvalueHistoryLength,
                                                                                 int ssaTrainingWindowSize,
                                                                                 int ssaSeasonalityWindowSize,
                                                                                 int confidence,
                                                                                 IEnumerable<DateIntegerDataViewModel> dataSet,
                                                                                 string detectionByColumnName,
                                                                                 bool detectedAlertsOnly = true)
        {
            return await Task.Run(() =>
            {
                var dataSetForMl = _mapper.Map<IEnumerable<DateData>>(dataSet.OrderBy(d => d.Date)).ToList();
                SpikesDetectedVector[] spikesDetected;
                spikesDetected = RunSpikeDetection(numericMethod, pvalueHistoryLength, ssaTrainingWindowSize, ssaSeasonalityWindowSize, confidence, detectionByColumnName, dataSetForMl);

                var result = new List<SpikeDetectionDataViewModel>(spikesDetected.Length);

                for (var i = 0; i < dataSetForMl.Count(); i++) //travers all dataset to find the same values, like detected spikes are, to copy a date 
                {
                    Func<double> roundUpUnsurness = () =>
                    {
                        return spikesDetected[i].Prediction[2] < 0.001 || Double.IsNaN(spikesDetected[i].Prediction[2])
                            ? 0
                            : Math.Round(spikesDetected[i].Prediction[2], 3);
                    };

                    if (detectedAlertsOnly && spikesDetected[i].Prediction[0] != 1) //1st (0) value in vector means if it is alert, or not (boolean 0,1)
                        continue;
                    {
                        var singleResult = CreateSingleResultMatchedWithSourceDataPoint<SpikeDetectionDataViewModel>(
                            sourceDataPoint: dataSetForMl[i],
                            detectedValue: spikesDetected[i].Prediction[1], // 2nd (1) = Score(value where the anomaly is detected e.g.number of sales)
                            isAlert: (int)spikesDetected[i].Prediction[0],  // Alert(0 for no alert, 1 for an alert)
                            unsurnessOrMag: roundUpUnsurness()); // 3rd (2) = P-value(value used to measure how likely an anomaly is to be true vs.background noise).

                        result.Add(singleResult);
                    }
                }
                //for (var i = 0; i < spikesDetected.Where(s => s.Prediction[0] == 1).Count(); i++) //lets take alerted ones only 
                //{
                //    result.Add(new SpikeDetectionDataViewModel()
                //    {
                //        Date = dataSetForMl.OrderBy(d => d.Date).LastOrDefault().Date.AddDays(7 * (i + 1)),
                //        IsAlert = true,
                //        ScoreOriginal = spikesDetected[i].Prediction[1],
                //        PValue = spikesDetected[i].Prediction[2]
                //    });
                //}
                return result;

                SpikesDetectedVector[] RunSpikeDetection(DetectionMethod numericMethod, int pvalueHistoryLength, int ssaTrainingWindowSize, int ssaSeasonalityWindowSize, int confidence, string detectionByColumnName, IEnumerable<DateData> dataSetForMl)
                {
                    switch (numericMethod)
                    {
                        case DetectionMethod.Iid:
                            return _spikesDetector.GetSpikesByIid<DateData>(pvalueHistoryLength,
                                                                            confidence,
                                                                            dataSetForMl,
                                                                            detectionByColumnName,
                                                                            AnomalySide.TwoSided)
                            .ToArray();
                        default:
                            return _spikesDetector.GetSpikesBySsa<DateData>(pvalueHistoryLength,
                                                                            ssaTrainingWindowSize,
                                                                            ssaSeasonalityWindowSize,
                                                                            confidence,
                                                                            dataSetForMl,
                                                                            detectionByColumnName,
                                                                            AnomalySide.TwoSided)
                            .ToArray();
                    }
                }


            });
        }

        private U CreateSingleResultMatchedWithSourceDataPoint<U>(DateData sourceDataPoint, double detectedValue, int isAlert, double unsurnessOrMag) where U : DateIntegerDataViewModel, new()
        {
            switch (new U())
            {
                //case WeeklyAmountChangePointsWithMissingDataResultDTO changePoints:
                //    changePoints.Items = items;
                //    changePoints.Sales = sales;
                //    changePoints.WeekendDate = weekendDate.ToShortDateString();
                //    changePoints.IsAlert = isAlert == 1;
                //    return changePoints as U;
                case SpikeDetectionDataViewModel spike:
                    spike.ScoreOriginal = detectedValue; //have to be exactly the same like a sourceDataPoint - matching is done before
                    spike.Date = sourceDataPoint.Date;
                    spike.IsAlert = isAlert == 1;
                    spike.PValue = unsurnessOrMag;
                    return spike as U;
                //case WeeklyAmountAnomalyResultDTO anomaly:
                //    anomaly.Items = items;
                //    anomaly.Sales = sales;
                //    anomaly.WeekendDate = weekendDate.ToShortDateString();
                //    anomaly.IsAlert = isAlert == 1;
                //    anomaly.Mag = unsurnessOrMag;
                //    return anomaly as U;
                default:
                    return null;
            }
        }
    }
}

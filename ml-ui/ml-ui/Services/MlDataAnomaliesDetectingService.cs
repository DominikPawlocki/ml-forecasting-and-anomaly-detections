using Microsoft.ML.Transforms.TimeSeries;
using ml_engine.AnomalyDetections;
using ml_data;
using ml_ui.ViewModels;
using System.Data;
using AutoMapper;

namespace ml_ui.Services
{
    public interface IMlDataAnomaliesDetectingService
    {
        Task<IEnumerable<SpikeDetectionDataViewModel>> DetectSpikes(DetectionMethod numericMethod,
                                                                    int pvalueHistoryLength,
                                                                    int ssaTrainingWindowSize,
                                                                    int ssaSeasonalityWindowSize,
                                                                    int confidence,
                                                                    IEnumerable<DateIntegerDataViewModel> dataSet,
                                                                    string detectionByColumnName,
                                                                    bool detectedAlertsOnly = true);
        public Task<IEnumerable<AnomalyDetectionDataViewModel>> DetectAnomalies(double threshold,
                                                                                int batchSize,
                                                                                double sensitivity,
                                                                                SrCnnDetectMode detectMode,
                                                                                int? period,
                                                                                SrCnnDeseasonalityMode deseasonalityMode,
                                                                                IEnumerable<DateIntegerDataViewModel> dataSet,
                                                                                string detectionByColumnName,
                                                                                bool detectedAlertsOnly = true);
    }

    public class MlDataAnomaliesDetectingService : IMlDataAnomaliesDetectingService
    {
        private readonly ISpikesDetector _spikesDetector;
        private readonly IAnomalyDetector _anomalyDetector;
        private readonly IMapper _mapper;

        public MlDataAnomaliesDetectingService(ISpikesDetector spikesDetector,
                                               IAnomalyDetector anomalyDetector,
                                               IMapper mapper)
        {
            _spikesDetector = spikesDetector;
            _anomalyDetector = anomalyDetector;
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
                SpikesDetectedVector[] spikesDetected = RunSpikeDetection(numericMethod, pvalueHistoryLength, ssaTrainingWindowSize, ssaSeasonalityWindowSize, confidence, detectionByColumnName, dataSetForMl);

                var result = new List<SpikeDetectionDataViewModel>(spikesDetected.Length);

                for (var i = 0; i < dataSetForMl.Count; i++) //travers all dataset to find the same values, like detected spikes are, to copy a date 
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
                            unsurnessOrMag: roundUpUnsurness(),// 3rd (2) = P-value(value used to measure how likely an anomaly is to be true vs.background noise).
                            expectedValue: 0); //Expected Value doesnt work for spike detection 
                        if (singleResult != null)
                            result.Add(singleResult);
                    }
                }
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

        public async Task<IEnumerable<AnomalyDetectionDataViewModel>> DetectAnomalies(double threshold,
                                                                                      int batchSize,
                                                                                      double sensitivity,
                                                                                      SrCnnDetectMode detectMode,
                                                                                      int? period,
                                                                                      SrCnnDeseasonalityMode deseasonalityMode,
                                                                                      IEnumerable<DateIntegerDataViewModel> dataSet,
                                                                                      string detectionByColumnName,
                                                                                      bool detectedAlertsOnly = true)
        {
            return await Task.Run(() =>
            {
                var dataSetForMl = _mapper.Map<IEnumerable<DateData>>(dataSet.OrderBy(d => d.Date)).ToList();
                AnomalyDetectedVector[] detectedAnomalies = _anomalyDetector.GetAnomalies<DateData>(detectionByColumnName,
                                                                                                    dataSetForMl,
                                                                                                    threshold,
                                                                                                    batchSize,
                                                                                                    sensitivity,
                                                                                                    detectMode,
                                                                                                    period,
                                                                                                    deseasonalityMode)
                .ToArray();

                var result = new List<AnomalyDetectionDataViewModel>(detectedAnomalies.Length);

                for (var i = 0; i < dataSetForMl.Count; i++) //travers all dataset to find the same values, like detected spikes are, to copy a date 
                {
                    Func<double> roundUpUnsurness = () =>
                    {
                        return detectedAnomalies[i].Prediction[2] < 0.001 || Double.IsNaN(detectedAnomalies[i].Prediction[2])
                            ? 0
                            : Math.Round(detectedAnomalies[i].Prediction[2], 3);
                    };
                    //The RawScore (2nd Index, [1] is output by SR to determine whether a point is an anomaly or not,
                    //under AnomalyAndMargin mode, when a point is an anomaly, an AnomalyScore will be calculated according to sensitivity setting.
                    if (detectedAlertsOnly && detectedAnomalies[i].Prediction[0] != 1)  //1st (0) value in vector means if it is alert, or not (boolean 0,1)
                        continue;
                    {
                        var singleResult = CreateSingleResultMatchedWithSourceDataPoint<AnomalyDetectionDataViewModel>(
                            sourceDataPoint: dataSetForMl[i],
                            detectedValue: detectedAnomalies[i].Prediction[1], // //The RawScore (2nd Index, [1] is output by SR to determine whether a point is an anomaly or not,
                            isAlert: (int)detectedAnomalies[i].Prediction[0],  // Alert(0 for no alert, 1 for an alert)
                            unsurnessOrMag: roundUpUnsurness(),
                            expectedValue: detectMode == SrCnnDetectMode.AnomalyAndExpectedValue ? detectedAnomalies[i].Prediction[3] : 0 //Only in AnomalyAndExpectedMode
                            );
                        if (singleResult != null)
                            result.Add(singleResult);
                    }
                }
                return result;
            });
        }

        private static U? CreateSingleResultMatchedWithSourceDataPoint<U>(DateData sourceDataPoint, double detectedValue, int isAlert, double unsurnessOrMag, double expectedValue) where U : DateIntegerDataViewModel, new()
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
                case AnomalyDetectionDataViewModel anomaly:
                    anomaly.ScoreOriginal = sourceDataPoint.Value;
                    anomaly.Date = sourceDataPoint.Date;
                    anomaly.IsAlert = isAlert == 1;
                    anomaly.Mag = unsurnessOrMag;
                    anomaly.ExpectedValue = expectedValue;
                    return anomaly as U;
                default:
                    return new U();
            }
        }
    }
}

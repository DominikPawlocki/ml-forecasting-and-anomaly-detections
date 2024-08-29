using Microsoft.ML.Transforms.TimeSeries;
using Microsoft.ML;
using Microsoft.ML.Data;
using ml_data;

namespace ml_engine.AnomalyDetections
{
    public interface ISpikesDetector
    {
        IEnumerable<SpikesDetectedVector> GetSpikesByIid<T>(int pvalueHistoryLength,
                                                            int confidence,
                                                            IEnumerable<DateData> data,
                                                            string detectionByColumnName,
                                                            AnomalySide side) where T : class;
        IEnumerable<SpikesDetectedVector> GetSpikesBySsa<T>(int pvalueHistoryLength,
                                                            int trainingWindowSize,
                                                            int seasonalityWindowSize,
                                                            int confidence,
                                                            IEnumerable<DateData> data,
                                                            string detectionByColumnName,
                                                            AnomalySide side) where T : class;
    }

    public class SpikesDetector : BaseForAllDetectors, ISpikesDetector
    {
        public IEnumerable<SpikesDetectedVector> GetSpikesByIid<T>(int pvalueHistoryLength,
                                                                   int confidence,
                                                                   IEnumerable<DateData> data,
                                                                   string detectionByColumnName,
                                                                   AnomalySide side = AnomalySide.TwoSided) where T : class
        {
            var orderedData = data.OrderBy(d => d.Date).ToList();
            var dataView = MlContext.Data.LoadFromEnumerable(orderedData);

            var estimatorChain = GetIidSpikesEstimator(pvalueHistoryLength, confidence, detectionByColumnName, side);

            var resultData = TransformModel<T>(dataView, estimatorChain);
            var spikesDetected = GetSpikes(resultData);

            return spikesDetected;
        }

        public IEnumerable<SpikesDetectedVector> GetSpikesBySsa<T>(int pvalueHistoryLength,
                                                                   int trainingWindowSize,
                                                                   int seasonalityWindowSize,
                                                                   int confidence,
                                                                   IEnumerable<DateData> data,
                                                                   string detectionByColumnName,
                                                                   AnomalySide side = AnomalySide.TwoSided) where T : class
        {
            var orderedData = data.OrderBy(d => d.Date).ToList();
            var dataView = MlContext.Data.LoadFromEnumerable(orderedData);

            var estimatorChain = GetSsaSpikesEstimator(pvalueHistoryLength,
                                                       trainingWindowSize,
                                                       seasonalityWindowSize,
                                                       confidence,
                                                       detectionByColumnName,
                                                       side);

            var resultData = TransformModel<T>(dataView, estimatorChain);
            var spikesDetected = GetSpikes(resultData);

            return spikesDetected;
        }

        private EstimatorChain<ITransformer> GetIidSpikesEstimator(int pvalHstLenght,
                                                                   int confidence,
                                                                   string detectionByColumnName,
                                                                   AnomalySide side = AnomalySide.TwoSided)
        {
            Console.WriteLine($"=============== Detect spikes with pattern IId ===============");
            var estimatorChain = new EstimatorChain<ITransformer>();

            return estimatorChain.Append(estimator: CreateIidSpikeEstimator(pvalHstLenght, confidence, detectionByColumnName, side));
        }

        private EstimatorChain<ITransformer> GetSsaSpikesEstimator(int pvalHstLenght,
                                                                   int tWinSize,
                                                                   int sWinSize,
                                                                   int confidence,
                                                                   string detectionByColumnName,
                                                                   AnomalySide side = AnomalySide.TwoSided)
        {
            Console.WriteLine($"=============== Detect spikes with pattern Ssa ===============");
            var estimatorChain = new EstimatorChain<ITransformer>();
            return estimatorChain.Append(estimator: CreateSsaSpikeEstimator(pvalHstLenght, tWinSize, sWinSize, confidence, detectionByColumnName, side));
        }

        private IEstimator<ITransformer> CreateIidSpikeEstimator(int pvalHstLenght,
                                                                 double confidence,
                                                                 string inputColumnName,
                                                                 AnomalySide side = AnomalySide.TwoSided)
        {
            return MlContext.Transforms.DetectIidSpike(
                    outputColumnName: nameof(SpikesDetectedVector.Prediction),
                    inputColumnName: inputColumnName,
                    confidence: confidence,
                    //https://github.com/dotnet/docs/issues/18394
                    pvalueHistoryLength: pvalHstLenght,
                    side: side);
        }

        private IEstimator<ITransformer> CreateSsaSpikeEstimator(int pvalHstLenght,
                                                                 int tWinSize,
                                                                 int sWinSize,
                                                                 double confidence,
                                                                 string inputColumnName,
                                                                 AnomalySide side = AnomalySide.TwoSided)
        {
            return MlContext.Transforms.DetectSpikeBySsa(outputColumnName: nameof(SpikesDetectedVector.Prediction),
                                                        inputColumnName: inputColumnName,
                                                        confidence: confidence,
                                                        pvalueHistoryLength: pvalHstLenght,
                                                        // The number of points from the beginning of the sequence used for training
                                                        trainingWindowSize: tWinSize,
                                                        // https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.timeseriescatalog.detectspikebyssa?view=ml-dotnet
                                                        // An upper bound on the largest relevant seasonality in the input time-series.
                                                        seasonalityWindowSize: sWinSize,
                                                        side: side);
        }

        private IEnumerable<SpikesDetectedVector> GetSpikes(IDataView resultData)
        {
            // Then, we invoke the detector and obtain a view of the output data.
            var spikesDetected = MlContext.Data.CreateEnumerable<SpikesDetectedVector>(resultData, reuseRowObject: false);
            return spikesDetected;
        }

    }
}

using Microsoft.ML.TimeSeries;
using Microsoft.ML.Transforms.TimeSeries;
using Microsoft.ML;
using Microsoft.ML.Data;
using ml_data;

namespace ml_engine.AnomalyDetections
{
    public interface IAnomalyDetector
    {
        IEnumerable<AnomalyDetectedVector> GetAnomalies<T>(string detectionByColumnName,
                                                           IEnumerable<DateData> data,
                                                           double threshold,
                                                           int batchSize,
                                                           double sensitivity,
                                                           SrCnnDetectMode detectMode,
                                                           int? period,
                                                           SrCnnDeseasonalityMode deseasonalityMode) where T : class;
    }

    public class AnomalyDetector : BaseForAllDetectors, IAnomalyDetector
    {
        public IEnumerable<AnomalyDetectedVector> GetAnomalies<T>(string detectionByColumnName,
                                                                  IEnumerable<DateData> data,
                                                                  double threshold,
                                                                  int batchSize,
                                                                  double sensitivity,
                                                                  SrCnnDetectMode detectMode,
                                                                  int? period,
                                                                  SrCnnDeseasonalityMode deseasonalityMode) where T : class
        {
            var orderedData = data.OrderBy(d => d.Date).ToList();
            var dataView = MlContext.Data.LoadFromEnumerable(orderedData);

            //STEP 1: Specify the input column and output column names.
            string outputColumnName = nameof(AnomalyDetectedVector.Prediction);

            Func<string, string> columnNameToDoubleTransformer = (string colName) => ($"{colName}Double");

            var estimatorChainTypeConversion = MlContext.Transforms.Conversion.
                    ConvertType(new[] {
                      new InputOutputColumnPair(columnNameToDoubleTransformer(detectionByColumnName), detectionByColumnName)
                    }, DataKind.Double);
            ITransformer tansformedModel = estimatorChainTypeConversion.Fit(CreateEmptyDataView<T>());
            var newIDataView = tansformedModel.Transform(dataView);

            if (period == null)
            {
                //STEP 2: Detect period on the given series.
                period = MlContext.AnomalyDetection.DetectSeasonality(newIDataView, columnNameToDoubleTransformer(detectionByColumnName));
            }
            if (period == -1)
            {
                //no period was able to be detected. Lets give it 0 then, cause it cannot be -1
                period = 0;
            }

            //STEP 3: Setup the parameters
            var options = new SrCnnEntireAnomalyDetectorOptions()
            {
                Threshold = threshold,
                Sensitivity = sensitivity,
                DetectMode = (Microsoft.ML.TimeSeries.SrCnnDetectMode)detectMode,
                Period = period.GetValueOrDefault(),
                BatchSize = batchSize,
                DeseasonalityMode = (Microsoft.ML.TimeSeries.SrCnnDeseasonalityMode)deseasonalityMode
            };

            //STEP 4: Invoke SrCnn algorithm to detect anomaly on the entire series.
            var outputDataView = MlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(newIDataView,
                                                                                       outputColumnName,
                                                                                       columnNameToDoubleTransformer(detectionByColumnName),
                                                                                       options);
            return GetAnomalies(outputDataView);
        }

        private IEnumerable<AnomalyDetectedVector> GetAnomalies(IDataView resultData)
        {
            return MlContext.Data.CreateEnumerable<AnomalyDetectedVector>(resultData, reuseRowObject: false);
        }
    }
}

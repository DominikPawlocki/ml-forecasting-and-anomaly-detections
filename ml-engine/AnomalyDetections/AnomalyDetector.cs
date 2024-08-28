using Microsoft.ML.TimeSeries;
using Microsoft.ML.Transforms.TimeSeries;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ml_engine.AnomalyDetections
{
    public interface IAnomalyDetector
    {
        //MLContext MlContext { get; }

        //IDataView GetChangePoints<T>(IDataView data,
        //                             DetectionMethod numericMethod,
        //                             string inputColumnName,
        //                             int confidence,
        //                             int changeHistoryLength,
        //                             int trainingWindowSize = 100,
        //                             int seasonalityWindowSize = 10,
        //                             ErrorFunction errorFunc = ErrorFunction.SignedDifference,
        //                             MartingaleType martingale = MartingaleType.Power,
        //                             double eps = 0.1) where T : class, new();
        IEnumerable<AnomalyDetectedVector> GetAnomalies<T>(string detectionByColumnName,
                                                           IDataView untrainedModel,
                                                           double threshold,
                                                           int batchSize,
                                                           double sensitivity,
                                                           SrCnnDetectMode detectMode,
                                                           int? period,
                                                           SrCnnDeseasonalityMode deseasonalityMode) where T : class, new();
    }

    public class AnomalyDetector : BaseForAllDetectors, IAnomalyDetector
    {

        public AnomalyDetector(MLContext machineLearningContext) : base(machineLearningContext)
        {
        }



        public IEnumerable<AnomalyDetectedVector> GetAnomalies<T>(string detectionByColumnName,
                                                                  IDataView untrainedModel,
                                                                  double threshold,
                                                                  int batchSize,
                                                                  double sensitivity,
                                                                  SrCnnDetectMode detectMode,
                                                                  int? period,
                                                                  SrCnnDeseasonalityMode deseasonalityMode) where T : class, new()
        {
            //STEP 1: Specify the input column and output column names.
            string outputColumnName = nameof(AnomalyDetectedVector.Prediction);

            //var estimatorChain = new EstimatorChain<ITransformer>();
            Func<string, string> columnNameToDoubleTransformer = (string colName) => ($"{colName}Double");

            var estimatorChainTypeConversion = MlContext.Transforms.Conversion.
                    ConvertType(new[] {
                      new InputOutputColumnPair(columnNameToDoubleTransformer(detectionByColumnName), detectionByColumnName)
                    }, DataKind.Double);
            ITransformer tansformedModel = estimatorChainTypeConversion.Fit(CreateEmptyDataView<T>());
            var newIDataView = tansformedModel.Transform(untrainedModel);


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
                DetectMode = (SrCnnDetectMode)detectMode, // Enum.Parse<SrCnnDetectMode>(detectMode.ToString(), true),
                Period = period.GetValueOrDefault(),
                BatchSize = batchSize,
                DeseasonalityMode = deseasonalityMode
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
            var result = MlContext.Data.CreateEnumerable<AnomalyDetectedVector>(resultData, reuseRowObject: false);
            return result;
        }

    }
}

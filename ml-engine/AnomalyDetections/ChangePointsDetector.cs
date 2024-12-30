using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using ml_data;

namespace ml_engine.AnomalyDetections
{
    public interface IChangePointsDetector
    {
        IEnumerable<SpikesDetectedVector> GetChangePoints<T>(string detectionByColumnName,
                                                             IEnumerable<DateData> data,
                                                             DetectionMethod numericMethod,
                                                             int confidence,
                                                             int changeHistoryLength,
                                                             int trainingWindowSize = 100,
                                                             int seasonalityWindowSize = 10,
                                                             ErrorFunction errorFunc = ErrorFunction.SignedDifference,
                                                             MartingaleType martingale = MartingaleType.Power,
                                                             double eps = 0.1) where T : class;
    }

    public class ChangePointsDetector : BaseForAllDetectors, IChangePointsDetector
    {
        public IEnumerable<SpikesDetectedVector> GetChangePoints<T>(string detectionByColumnName,
                                                                    IEnumerable<DateData> data,
                                                                    DetectionMethod numericMethod,
                                                                    int confidence,
                                                                    int changeHistoryLength,
                                                                    int trainingWindowSize = 100,
                                                                    int seasonalityWindowSize = 10,
                                                                    ErrorFunction errorFunc = ErrorFunction.SignedDifference,
                                                                    MartingaleType martingale = MartingaleType.Power,
                                                                    double eps = 0.1) where T : class
        {
            var orderedData = data.OrderBy(d => d.Date).ToList();
            var dataView = MlContext.Data.LoadFromEnumerable(orderedData);

            //STEP 1: Specify the input column and output column names.
            string outputColumnName = nameof(AnomalyDetectedVector.Prediction);

            Func<string, string> columnNameToDoubleTransformer = (string colName) => ($"{colName}Double");

            var chain = new EstimatorChain<ITransformer>().Append(
                MlContext.Transforms.Conversion.ConvertType(new[] {
                      new InputOutputColumnPair(columnNameToDoubleTransformer(detectionByColumnName), detectionByColumnName)
                    }, DataKind.Single)
                );

            var chain2 = numericMethod == DetectionMethod.Ssa
                ? chain.Append(estimator: CreateSsaChangePointEstimator(outputColumnName, columnNameToDoubleTransformer(detectionByColumnName), confidence, changeHistoryLength,
                                                                        trainingWindowSize, seasonalityWindowSize,
                                                                        errorFunc, martingale, eps))
                : chain.Append(estimator: CreateIidChangePointEstimator(outputColumnName, columnNameToDoubleTransformer(detectionByColumnName), confidence,
                                                                        changeHistoryLength, martingale, eps));
            // STEP 2:The Transformed Model.
            // In IID Spike detection, we don't need to do training, we just need to do transformation. 
            // As you are not training the model, there is no need to load IDataView with real data, you just need schema of data.
            // So create empty data view and pass to Fit() method. 
            ITransformer tansformedModel = chain2.Fit(CreateEmptyDataView<T>());

            // STEP 3: Use/test model. Apply data transformation to create predictions.
            IDataView newIDataView = tansformedModel.Transform(dataView);

            return GetChangePoints(newIDataView);
        }

        private IEstimator<ITransformer> CreateIidChangePointEstimator(string outputColumnName,
                                                                       string inputColumnName,
                                                                       double confidence,
                                                                       int changeHistoryLength,
                                                                       MartingaleType martingale = MartingaleType.Power,
                                                                       double eps = 0.1)
        {
            return MlContext.Transforms.DetectIidChangePoint(outputColumnName: outputColumnName,
                                                             inputColumnName: inputColumnName,
                                                             confidence: confidence,
                                                             changeHistoryLength: changeHistoryLength,
                                                             martingale: martingale,
                                                             eps: eps);
        }

        private IEstimator<ITransformer> CreateSsaChangePointEstimator(string outputColumnName,
                                                                       string inputColumnName,
                                                                       double confidence,
                                                                       int changeHistoryLength,
                                                                       int trainingWindowSize = 100,
                                                                       int seasonalityWindowSize = 10,
                                                                       ErrorFunction errorFunc = ErrorFunction.SignedDifference,
                                                                       MartingaleType martingale = MartingaleType.Power,
                                                                       double eps = 0.1)
        {
            return MlContext.Transforms.DetectChangePointBySsa(outputColumnName: outputColumnName,
                                                               inputColumnName: inputColumnName,
                                                               confidence: confidence,
                                                               changeHistoryLength: changeHistoryLength,
                                                               trainingWindowSize: trainingWindowSize,
                                                               seasonalityWindowSize: seasonalityWindowSize,
                                                               errorFunction: errorFunc,
                                                               martingale: martingale,
                                                               eps: eps);
        }

        //SpikesDetectorVector can be reused, as its the same for changepoints
        private IEnumerable<SpikesDetectedVector> GetChangePoints(IDataView resultData)
        {
            return MlContext.Data.CreateEnumerable<SpikesDetectedVector>(resultData, reuseRowObject: false);
        }
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace ml_engine.AnomalyDetections
{
    public interface IChangePointsDetector
    {
        IEnumerable<SpikesDetectedVector> GetChangePoints<T>(IDataView data,
                                                             DetectionMethod numericMethod,
                                                             string inputColumnName,
                                                             int confidence,
                                                             int changeHistoryLength,
                                                             int trainingWindowSize = 100,
                                                             int seasonalityWindowSize = 10,
                                                             ErrorFunction errorFunc = ErrorFunction.SignedDifference,
                                                             MartingaleType martingale = MartingaleType.Power,
                                                             double eps = 0.1) where T : class, new();
    }

    public class ChangePointsDetector : BaseForAllDetectors, IChangePointsDetector
    {
        public IEnumerable<SpikesDetectedVector> GetChangePoints<T>(IDataView data,
                                                                    DetectionMethod numericMethod,
                                                                    string inputColumnName,
                                                                    int confidence,
                                                                    int changeHistoryLength,
                                                                    int trainingWindowSize = 100,
                                                                    int seasonalityWindowSize = 10,
                                                                    ErrorFunction errorFunc = ErrorFunction.SignedDifference,
                                                                    MartingaleType martingale = MartingaleType.Power,
                                                                    double eps = 0.1) where T : class, new()
        {
            var transformedModel = GetChangePointTransformedModel<T>(data,
                                                                     numericMethod,
                                                                     inputColumnName,
                                                                     confidence,
                                                                     changeHistoryLength,
                                                                     trainingWindowSize,
                                                                     seasonalityWindowSize,
                                                                     errorFunc,
                                                                     martingale);

            var resultData = GetChangePoints(transformedModel);

            return resultData;
        }

        private IDataView GetChangePointTransformedModel<T>(IDataView data,
                                                            DetectionMethod numericMethod,
                                                            string inputColumnName,
                                                            int confidence,
                                                            int changeHistoryLength,
                                                            int trainingWindowSize = 100,
                                                            int seasonalityWindowSize = 10,
                                                            ErrorFunction errorFunc = ErrorFunction.SignedDifference,
                                                            MartingaleType martingale = MartingaleType.Power,
                                                            double eps = 0.1)
           where T : class, new()
        {
            Console.WriteLine($"===============Detect changePoint with pattern {numericMethod} ===============");
            // IndicateMissingValues is used to create a boolean containing 'true'
            // where the value in the input column is missing. For floats and doubles, missing values are NaN. We can use an array of
            // InputOutputColumnPair to apply the MissingValueIndicatorEstimator to multiple columns in one pass over the data.
            var chain = new EstimatorChain<ITransformer>().Append(MlContext.Transforms.IndicateMissingValues(new[] {
                new InputOutputColumnPair(nameof(ChangepointsWithMissingValuesDetectedVector.IsMissing), inputColumnName)
            }));

            //this estimator makes a chances for detection a missing value as alert higher, cause for any driver which has non-zero values,
            //the zero value which is "inserted" here should be detected as alert.
            /*  var chain = new EstimatorChain<ITransformer>().Append(estimator: MlContext.Transforms.ReplaceMissingValues(
            new[] { new InputOutputColumnPair(inputColumnName) }, Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.DefaultValue));
            */
            var chain2 = numericMethod == DetectionMethod.Ssa
                ? chain.Append(estimator: CreateSsaChangePointEstimator(inputColumnName, confidence, changeHistoryLength,
                                                                        trainingWindowSize, seasonalityWindowSize,
                                                                        errorFunc, martingale, eps))
                : chain.Append(estimator: CreateIidChangePointEstimator(inputColumnName, confidence,
                                                                        changeHistoryLength, martingale, eps));
            // STEP 2:The Transformed Model.
            // In IID Spike detection, we don't need to do training, we just need to do transformation. 
            // As you are not training the model, there is no need to load IDataView with real data, you just need schema of data.
            // So create empty data view and pass to Fit() method. 
            ITransformer tansformedModel = chain2.Fit(CreateEmptyDataView<T>());

            // STEP 3: Use/test model. Apply data transformation to create predictions.
            IDataView transformedData = tansformedModel.Transform(data);
            return transformedData;
        }

        private IEstimator<ITransformer> CreateIidChangePointEstimator(string inputColumnName,
                                                               double confidence,
                                                               int changeHistoryLength,
                                                               MartingaleType martingale = MartingaleType.Power,
                                                               double eps = 0.1)
        {
            return MlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(SpikesDetectedVector.Prediction),
                                                             inputColumnName: inputColumnName,
                                                             confidence: confidence,
                                                             changeHistoryLength: changeHistoryLength,
                                                             martingale: martingale,
                                                             eps: eps);
        }

        private IEstimator<ITransformer> CreateSsaChangePointEstimator(string inputColumnName,
                                                                       double confidence,
                                                                       int changeHistoryLength,
                                                                       int trainingWindowSize = 100,
                                                                       int seasonalityWindowSize = 10,
                                                                       ErrorFunction errorFunc = ErrorFunction.SignedDifference,
                                                                       MartingaleType martingale = MartingaleType.Power,
                                                                       double eps = 0.1)
        {
            return MlContext.Transforms.DetectChangePointBySsa(outputColumnName: nameof(SpikesDetectedVector.Prediction),
                                                               inputColumnName: inputColumnName,
                                                               confidence: confidence,
                                                               changeHistoryLength: changeHistoryLength,
                                                               trainingWindowSize: trainingWindowSize,
                                                               seasonalityWindowSize: seasonalityWindowSize,
                                                               errorFunction: errorFunc,
                                                               martingale: martingale,
                                                               eps: eps);
        }

        private IEnumerable<ChangepointsWithMissingValuesDetectedVector> GetChangePointsAndMissingData(IDataView resultData)
        {
            return MlContext.Data.CreateEnumerable<ChangepointsWithMissingValuesDetectedVector>(resultData, reuseRowObject: false);
        }

        //SpikesDetectorVector can be reused, as its the same for changepoints
        private IEnumerable<SpikesDetectedVector> GetChangePoints(IDataView resultData)
        {
            return MlContext.Data.CreateEnumerable<SpikesDetectedVector>(resultData, reuseRowObject: false);
        }
    }
}

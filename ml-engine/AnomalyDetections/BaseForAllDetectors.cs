using Microsoft.ML;
using Microsoft.ML.Data;

namespace ml_engine.AnomalyDetections
{
    public class BaseForAllDetectors
    {
        public MLContext MlContext { get; }

        public BaseForAllDetectors()
        {
            MlContext = new MLContext(1);
        }

        protected IDataView CreateEmptyDataView<T>() where T : class
        {
            //Create empty DataView. We just need the schema to call fit() 
            return MlContext.Data.LoadFromEnumerable(new List<T>());
        }

        protected IDataView TransformModel<T>(IDataView untrainedModel, EstimatorChain<ITransformer> estimatorChain) where T : class
        {
            // STEP 2:The Transformed Model.
            // For these detections, don't need to do training, we just need to do transformation. 
            // As you are not training the model, there is no need to load IDataView with real data, you just need schema of data.
            // So create empty data view and pass to Fit() method. 
            ITransformer tansformedModel = estimatorChain.Fit(CreateEmptyDataView<T>());

            // STEP 3: Use/test model. Apply data transformation to create predictions.
            IDataView transformedData = tansformedModel.Transform(untrainedModel);

            return transformedData;
        }
    }
}

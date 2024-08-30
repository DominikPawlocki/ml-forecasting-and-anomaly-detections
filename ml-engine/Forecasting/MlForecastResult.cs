using Microsoft.ML.Data;

namespace ml_engine.Forecasting
{
    public class MlForecastResult
    {
        public MlForecastResult()
        {
            Predictions = [];
            ConfidenceLowerBounds = [];
            ConfidenceUpperBounds = [];
        }

        [VectorType(3)]
        public float[] Predictions { get; set; }
        [VectorType(3)]
        public float[] ConfidenceLowerBounds { get; set; }
        [VectorType(3)]
        public float[] ConfidenceUpperBounds { get; set; }
    }
}

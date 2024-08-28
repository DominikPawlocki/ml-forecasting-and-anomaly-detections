using Microsoft.ML.Data;

namespace ml_engine
{
    public class MlForecastResult
    {
        [VectorType(3)]
        public float[] Predictions { get; set; }
        [VectorType(3)]
        public float[] ConfidenceLowerBounds { get; set; }
        [VectorType(3)]
        public float[] ConfidenceUpperBounds { get; set; }
    }
}

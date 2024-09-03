using Microsoft.ML.Data;

namespace ml_engine.Forecasting
{
    public class MlSSAPrediction
    {
        public MlSSAPrediction()
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

    public class MlLinearRegressionDateValuePrediction
    {
        public DateTime Date;
        public float DateEncoded;
        [ColumnName("Score")]
        public float Value;
    }
}

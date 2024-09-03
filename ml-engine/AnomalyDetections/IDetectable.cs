using Microsoft.ML.Data;

namespace ml_engine.AnomalyDetections
{
    public interface IDetectable
    {
        double[] Prediction { get; }
    }

    public interface IMissingValueDetectable
    {
        bool IsMissing { get; set; }
    }

    /// <summary>
    /// Vector of System.Double. Each Prediction in `predictions` returns back a vector containing three values:
    /// 0 = Alert(0 for no alert, 1 for an alert)
    /// 1 = Score(value where the anomaly is detected e.g.number of sales)
    /// 2 = P-value(value used to measure how likely an anomaly is to be true vs.background noise).
    /// P-value is a metric between zero and one. The lower the value, the larger the probability that we’re looking at a spike.
    /// </summary>
    public class SpikesDetectedVector : IDetectable
    {
        public SpikesDetectedVector()
        {
            Prediction = [];
        }
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }

    /// <summary>
    /// Vector to hold anomaly detection results. 
    /// AnomalyOnly: isAnomaly, rawScore, Mag
    /// AnomalyAndExpectedValue: isAnomaly, rawScore, Mag, ExpectedValue
    /// AnomalyAndMargin: isAnomaly, anomalyScore, magnitude, expectedValue, boundaryUnits, upperBoundary and lowerBoundary
    /// </summary>
    public class AnomalyDetectedVector : IDetectable
    {
        [VectorType(7)]
        public double[] Prediction { get; set; } = [];
    }

    public class ChangepointsWithMissingValuesDetectedVector : IDetectable, IMissingValueDetectable
    {
        [VectorType(3)]
        public double[] Prediction { get; set; } = [];

        public bool IsMissing { get; set; }
    }
}

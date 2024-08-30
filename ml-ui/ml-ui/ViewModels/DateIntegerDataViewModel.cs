using Microsoft.ML.Data;

namespace ml_ui.ViewModels
{
    public class DateIntegerDataViewModel
    {
        public DateTime Date { get; set; }
        public int Value { get; set; }
    }

    public class DateIntegerForecasterDataViewModel : DateIntegerDataViewModel
    {
        public bool IsForecasted { get; set; }
        public float ConfidenceLowerBound { get; set; }
        [VectorType(3)]
        public float ConfidenceUpperBound { get; set; }
    }

    /// <summary>
    /// An output VM
    /// Vector of System.Double goes out from ML library. Each Prediction in `predictions` returns back a vector containing three values:
    /// 0 = Alert(0 for no alert, 1 for an alert)
    /// 1 = Score(value where the anomaly is detected e.g.number of sales)
    /// 2 = P-value(value used to measure how likely an anomaly is to be true vs.background noise).
    /// P-value is a metric between zero and one. The lower the value, the larger the probability that we’re looking at a spike.
    /// </summary>
    public class SpikeDetectionDataViewModel : DateIntegerDataViewModel
    {
        public bool IsAlert { get; set; }
        public double ScoreOriginal { get; set; }

        public new int Value { get { return (int)ScoreOriginal; } }
        public double PValue { get; set; }
    }

    /// <param name="detectMode">An enum type of <see cref="SrCnnDetectMode"/>.
    /// When set to AnomalyOnly, the output vector would be a 3-element Double vector of (IsAnomaly, RawScore, Mag).
    /// When set to AnomalyAndExpectedValue, the output vector would be a 4-element Double vector of (IsAnomaly, RawScore, Mag, ExpectedValue).
    /// When set to AnomalyAndMargin, the output vector would be a 7-element Double vector of (IsAnomaly, AnomalyScore, Mag, ExpectedValue, BoundaryUnit, UpperBoundary, LowerBoundary).
    /// The RawScore is output by SR to determine whether a point is an anomaly or not, under AnomalyAndMargin mode, when a point is an anomaly, an AnomalyScore will be calculated according to sensitivity setting.
    /// Default value is AnomalyOnly.</param>
    public class AnomalyDetectionDataViewModel : DateIntegerDataViewModel
    {
        public bool IsAlert { get; set; }
        public double ScoreOriginal { get; set; }
        public new int Value { get { return (int)ScoreOriginal; } }

        //public double RawScore { get; set; } --> not needed for me now, its internal numeric method value output, cant find valueable meaning for me
        public double Mag { get; set; }
        public double ExpectedValue { get; set; }
    }


}

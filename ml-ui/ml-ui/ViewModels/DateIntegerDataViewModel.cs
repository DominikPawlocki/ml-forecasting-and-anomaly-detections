using Microsoft.ML.Data;

namespace ml_ui.ViewModels
{
    public class DateIntegerDataViewModel : PropertyChangedNotifier
    {
        private DateTime _date;
        private int _val;

        public DateTime Date
        {
            get { return _date; }
            set
            {
                if (_date != value)
                {
                    _date = value;
                    NotifyPropertyChanged(nameof(this.Date));
                }
            }
        }

        public int Value { get => _val; set => _val = value; }
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


}

using static System.Runtime.InteropServices.JavaScript.JSType;

namespace ml_ui.ViewModels
{
    public class AnomalyDetectionViewModel : PropertyChangedNotifier
    {
        private IEnumerable<DateIntegerDataViewModel> _data;

        public IEnumerable<DateIntegerDataViewModel> Data
        {
            get { return _data; }
            set
            {
                if (_data != value)
                {
                    _data = value;
                    NotifyPropertyChanged(nameof(this.Data));
                }
            }
        }

        public IEnumerable<DateIntegerDataViewModel> DataWithAnomaliesDetected
        {
            get { return _data != null ? _data.Where(d => d.IsAnomaly) : []; }
        }
    }
}

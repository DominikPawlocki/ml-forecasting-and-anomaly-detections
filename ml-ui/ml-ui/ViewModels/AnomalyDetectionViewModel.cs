namespace ml_ui.ViewModels
{
    public class AnomalyDetectionViewModel : ViewModelBase
    {
        public IEnumerable<DateIntegerDataViewModel> DataPointsDetectedAsAnomalies
        {
            get { return _data != null ? _data.Where(d => d.IsAnomaly) : []; }
        }
    }
}

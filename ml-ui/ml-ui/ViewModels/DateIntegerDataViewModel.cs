namespace ml_ui.ViewModels
{
    public class DateIntegerDataViewModel : PropertyChangedNotifier
    {
        private DateTime _date;
        private int _val;
        private bool _isAnomaly;

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


        public bool IsAnomaly { get => _isAnomaly; set => _isAnomaly = value; }
    }
}

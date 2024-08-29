namespace ml_ui.ViewModels
{
    public abstract class ViewModelBase : PropertyChangedNotifier
    {
        public string? ErrorOccuredText { get; set; }
        public bool ShowError { get; set; }

        protected IEnumerable<DateIntegerDataViewModel> _data;

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

        public bool ShowDataLabels { get; set; } = false;
        public int HowManyToGenerate { get; set; } = 20;
        public int RandomUpperBound { get; set; } = 20;
        public int RandomLowerBound { get; set; } = -20;
        public int RandomDiscrepanciesAmount { get; set; } = 2;
        public int LinearDiscrepancy { get; set; } = 2;
        public int SinusDiscrepancy { get; set; } = 3;
    }
}

namespace ml_ui.ViewModels
{
    public abstract class ViewModelBase : PropertyChangedNotifier
    {
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
    }
}

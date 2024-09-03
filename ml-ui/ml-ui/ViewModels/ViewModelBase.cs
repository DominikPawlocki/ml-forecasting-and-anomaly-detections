namespace ml_ui.ViewModels
{
    public abstract class ViewModelBase
    {
        public string? ErrorOccuredText { get; set; }
        public bool ShowError { get; set; }

        public IEnumerable<DateIntegerDataViewModel> Data { get; set; } = [];

        public bool ShowDataLabels { get; set; } = false;
        public int HowManyToGenerate { get; set; } = 20;
        public int RandomUpperBound { get; set; } = 20;
        public int RandomLowerBound { get; set; } = -20;
        public int RandomDiscrepanciesAmount { get; set; } = 2;
        public int LinearDiscrepancy { get; set; } = 2;
        public int SinusDiscrepancy { get; set; } = 3;
    }
}

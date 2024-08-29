using ml_engine.AnomalyDetections;

namespace ml_ui.ViewModels
{
    public class AnomalyDetectionViewModel : ViewModelBase
    {
        // --------- SPIKES -----------
        public DetectionMethod Spikes_NumericMethod { get; set; }
        /// <summary>
        /// The size of the sliding window for computing the p-value.</param>
        /// </summary>
        public int Spikes_PValueHistoryLength { get; set; }
        /// <summary>
        /// The number of points from the beginning of the sequence used for training.
        /// </summary>
        public int Spikes_TrainingWindowSize { get; set; }
        /// <summary>
        /// An upper bound on the largest relevant seasonality in the input time-series.
        /// </summary>
        public int Spikes_SeasonalityWindowSize { get; set; }
        public int Spikes_Confidence { get; set; }
        // ------------- ANOMALIES -------------
        public double Anomalies_Threshold { get; set; }
        public int Anomalies_BatchSize { get; set; }
        public double Anomalies_Sensitivity { get; set; }
        public SrCnnDetectMode Anomalies_DetectMode { get; set; }
        public int Anomalies_Period { get; set; }
        public SrCnnDeseasonalityMode Anomalies_DeseasonalityMode { get; set; }
        //---------- CHANGEPOINTS --------------
        public DetectionMethod ChangePoints_NumericMethod { get; set; }
        public int ChangePoints_Confidence { get; set; }
        public int ChangePoints_ChangeHistoryLength { get; set; }
        public int ChangePoints_TrainingWindowSize { get; set; }
        public int ChangePoints_SeasonalityWindowSize { get; set; }


        //public IEnumerable<DateIntegerDataViewModel> DataPointsDetectedAsAnomalies
        //{
        //    get { return _data != null ? _data.Where(d => d.IsAnomaly) : []; }
        //}

        public IEnumerable<SpikeDetectionDataViewModel>? SpikesDetected
        {
            get; set;
        }

        private void ClearModelSpikes()
        {
            SpikesDetected = new List<SpikeDetectionDataViewModel>(0);
        }

        internal void SetUpDefaults()
        {
            ShowError = false;
            ClearModelSpikes();
            SetDefaultModelParametersAccordingtoDataSetForSpikes();
        }

        private void SetDefaultModelParametersAccordingtoDataSetForSpikes()
        {
            Spikes_PValueHistoryLength = Data.Count() / 2;
            Spikes_TrainingWindowSize = Data.Count();
            Spikes_SeasonalityWindowSize = Data.Count() / 8;
            Spikes_Confidence = 99;
        }

    }
}

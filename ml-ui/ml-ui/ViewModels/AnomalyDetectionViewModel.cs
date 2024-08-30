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
        public int? Anomalies_Period { get; set; }
        public SrCnnDeseasonalityMode Anomalies_DeseasonalityMode { get; set; }
        //---------- CHANGEPOINTS --------------
        public DetectionMethod ChangePoints_NumericMethod { get; set; }
        public int ChangePoints_Confidence { get; set; }
        public int ChangePoints_ChangeHistoryLength { get; set; }
        public int ChangePoints_TrainingWindowSize { get; set; }
        public int ChangePoints_SeasonalityWindowSize { get; set; }

        public IEnumerable<SpikeDetectionDataViewModel>? SpikesDetected
        {
            get; set;
        }
        public IEnumerable<AnomalyDetectionDataViewModel>? AnomaliesDetected
        {
            get; set;
        }

        internal void ClearModelSpikes()
        {
            SpikesDetected = new List<SpikeDetectionDataViewModel>(0);
        }
        internal void ClearModelAnomalies()
        {
            AnomaliesDetected = new List<AnomalyDetectionDataViewModel>(0);
        }

        internal void SetUpDefaults()
        {
            ShowError = false;
            ClearModelSpikes();
            ClearModelAnomalies();
            SetDefaultModelParametersAccordingtoDataSetForSpikes();
            SetDefaultModelParametersAccordingtoDataSetForAnomalies();
        }

        private void SetDefaultModelParametersAccordingtoDataSetForSpikes()
        {
            Spikes_PValueHistoryLength = Data.Count() / 2;
            Spikes_TrainingWindowSize = Data.Count();
            Spikes_SeasonalityWindowSize = Data.Count() / 8;
            Spikes_Confidence = 99;
        }

        /// <param name="threshold">The threshold to determine an anomaly. An anomaly is detected when the calculated SR raw score for a given point is more than the set threshold. This threshold must  fall between [0,1], and its default value is 0.3.</param>
        /// <param name="batchSize">Divide the input data into batches to fit srcnn model.
        /// When set to -1, use the whole input to fit model instead of batch by batch, when set to a positive integer, use this number as batch size.
        /// Must be -1 or a positive integer no less than 12. Default value is 1024.</param>
        /// <param name="sensitivity">Sensitivity of boundaries, only useful when srCnnDetectMode is AnomalyAndMargin. Must be in [0,100]. Default value is 99.</param>
        /// <param name="detectMode">An enum type of <see cref="SrCnnDetectMode"/>.
        private void SetDefaultModelParametersAccordingtoDataSetForAnomalies()
        {
            Anomalies_BatchSize = -1; //Must be -1 or a positive integer no less than 12. Default value is 1024
            Anomalies_DeseasonalityMode = SrCnnDeseasonalityMode.Stl;
            Anomalies_DetectMode = SrCnnDetectMode.AnomalyOnly;
            Anomalies_Period = null; //when setting null, system tries to detect seasonability of data itself.
            Anomalies_Sensitivity = 99; //Sensitivity of boundaries, only useful when srCnnDetectMode is AnomalyAndMargin.
            Anomalies_Threshold = 0.3; //This threshold must  fall between [0,1], and its default value is 0.3
        }
    }
}

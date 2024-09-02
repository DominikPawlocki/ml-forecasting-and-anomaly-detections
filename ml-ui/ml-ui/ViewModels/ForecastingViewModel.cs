using Microsoft.ML.Data;
using Microsoft.ML;

namespace ml_ui.ViewModels
{
    public class ForecastingViewModel : ViewModelBase
    {
        public IEnumerable<DateIntegerForecasterDataViewModel>? DataPointsPredicted { get; set; } = [];
        public IEnumerable<DateIntegerForecasterDataViewModel>? RegressionTrainedModelDataOutput { get; set; } = [];
        public TransformerChain<ITransformer>? TrainedRegressionModel { get; set; }

        public int SeriesLenght { get; set; }

        public int WindowSize { get; set; } //SSA
        public int TrainSize { get; set; } //SSA

        public bool ShowTrainedModel { get; set; }

        public string RegressionLearner { get; set; } = "SDCA";

        private void ClearModel()
        {
            DataPointsPredicted = [];
            RegressionTrainedModelDataOutput = [];
            TrainedRegressionModel = null;
        }

        internal void SetUpDefaults()
        {
            ShowError = false;
            ClearModel();
            (WindowSize, TrainSize, SeriesLenght) = SetDefaultModelTrainingParametersAccordingtoDataSet();
        }

        private (int winSize, int trnSize, int serLen) SetDefaultModelTrainingParametersAccordingtoDataSet()
        {
            // this values are very important cause its defines how algorithm works. In the method GetForecastingPipeline you can find it explained.
            int trnSize = Data.Count();       //'The input size for training should be greater than twice the window size.'
            int serLen = Data.Count() / 8;    //'The series length should be greater than the window size.
            int winSize = serLen - 2; //as minumum WindowSize is 2, then it makes that minumum actualData is 32.

            //Trying to load 2 years of data by default. However it might be that there is no so much data available, or there are different cases
            //In that case, the parameters has to be adjusted cause it will predict 0 or gives exception
            return Data.Count() switch
            {
                0 => (0, 0, 0),
                // --> we have not much data and default params wouldnt work. let adjust
                int a when a < 8 => (2, trnSize, 3),
                // --> we have not much data and default params wouldnt work. let adjust
                int a when a < 20 => (2, trnSize, 4),
                // --> we have not much data and default params wouldnt work. let adjust
                int a when a < 32 => (6, trnSize, 10),
                _ => (winSize, trnSize, serLen),
            };
        }
    }
}

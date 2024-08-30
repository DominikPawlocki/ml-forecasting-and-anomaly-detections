﻿namespace ml_ui.ViewModels
{
    public class ForecastingViewModel : ViewModelBase
    {
        public IEnumerable<DateIntegerForecasterDataViewModel>? DataPointsPredicted
        {
            get; set;
        }

        public int SeriesLenght { get; set; }

        public int WindowSize { get; set; }
        public int TrainSize { get; set; }

        private void ClearModel()
        {
            DataPointsPredicted = new List<DateIntegerForecasterDataViewModel>(0);
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
            switch (Data.Count())
            {
                case 0:
                    return (0, 0, 0);
                case int a when a < 8: // --> we have not much data and default params wouldnt work. let adjust
                    return (2, trnSize, 3);
                case int a when a < 20: // --> we have not much data and default params wouldnt work. let adjust
                    return (2, trnSize, 4);
                case int a when a < 32: // --> we have not much data and default params wouldnt work. let adjust
                    return (6, trnSize, 10);
                default:
                    return (winSize, trnSize, serLen);
            }
        }
    }
}

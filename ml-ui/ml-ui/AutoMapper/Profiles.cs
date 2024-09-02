using AutoMapper;
using ml_data;
using ml_engine.Forecasting;
using ml_ui.ViewModels;

namespace ml_ui.AutoMapper
{
    public class Profiles : Profile
    {
        public Profiles()
        {
            CreateMap<DateData, DateIntegerDataViewModel>();
            CreateMap<DateIntegerDataViewModel, DateData>().ConstructUsing((src, res) =>
            {
                return new DateData(src.Date, src.Value);
            });
            CreateMap<MlLinearRegressionDateValuePredition, DateIntegerForecasterDataViewModel>();
        }
    }
}
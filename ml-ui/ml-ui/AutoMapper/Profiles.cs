using AutoMapper;
using ml_data;
using ml_ui.ViewModels;

namespace ml_ui.AutoMapper
{
    public class Profiles : Profile
    {
        public Profiles()
        {
            CreateMap<DateData, DateIntegerDataViewModel>();
        }
    }
}

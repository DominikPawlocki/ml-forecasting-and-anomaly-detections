using AutoMapper;
using ml_data;
using ml_ui.ViewModels;

namespace ml_ui
{
    public class WebSocketDataConnector(IDataGenerator provider, IMapper mapper)
    {
        //private readonly IDatabaseContextService _databaseContextService; --> for example, add database provider here
        private readonly IDataGenerator _dataProvider = provider;
        private readonly IMapper _mapper = mapper;

        public async Task<IEnumerable<DateIntegerDataViewModel>> GetIntegerRandomData(int howManyToGenerate,
                                                                                      int upperBound,
                                                                                      int lowerBound,
                                                                                      int randomDiscrepanciesAmount)
        {
            return await Task.Run(() =>
               {
                   var integers = _dataProvider.GenerateRandomIntegersDataSetWithDateIndexedWeekly(howManyToGenerate,
                                                                                                   new DateTime(2024, 01, 01),
                                                                                                   lowerBound,
                                                                                                   upperBound,
                                                                                                   randomDiscrepanciesAmount);
                   return _mapper.Map<IEnumerable<DateIntegerDataViewModel>>(integers);
               });
        }

        public async Task<IEnumerable<DateIntegerDataViewModel>> GetLinearData(int howManyToGenerate, float linearDiscrepancy)
        {
            return await Task.Run(() =>
            {
                var integers = _dataProvider.GenerateLinearDataSetWithDateIndexedWeekly(howManyToGenerate, new DateTime(2024, 01, 01), linearDiscrepancy);
                return _mapper.Map<IEnumerable<DateIntegerDataViewModel>>(integers);
            });
        }

        public async Task<IEnumerable<DateIntegerDataViewModel>> GetSinusData(int howManyToGenerate, int upperBound, float sinusDiscrepancy)
        {
            return await Task.Run(() =>
            {
                var integers = _dataProvider.GenerateSinusDataSetWithDateIndexedWeekly(howManyToGenerate, new DateTime(2024, 01, 01), upperBound, sinusDiscrepancy);
                return _mapper.Map<IEnumerable<DateIntegerDataViewModel>>(integers);
            });
        }
    }
}
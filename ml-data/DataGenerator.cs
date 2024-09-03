namespace ml_data
{
    public interface IDataGenerator
    {
        IEnumerable<DateData> GenerateRandomIntegersDataSetWithDateIndexedWeekly(int howMany,
                                                                                DateTime startingDate,
                                                                                int valueLowerBound,
                                                                                int valueHigherBound,
                                                                                int randomDiscrepanciesAmount);
        IEnumerable<DateData> GenerateLinearDataSetWithDateIndexedWeekly(int howMany,
                                                                         DateTime startingDate,
                                                                         float linearDiscrepancy);
        IEnumerable<DateData> GenerateSinusDataSetWithDateIndexedWeekly(int howMany,
                                                                        DateTime startingDate,
                                                                        int valueHigherBound,
                                                                        float sinusDiscrepancy);
    }

    public class DataGenerator : IDataGenerator
    {
        public IEnumerable<DateData> GenerateRandomIntegersDataSetWithDateIndexedWeekly(int howMany,
                                                                                        DateTime startingDate,
                                                                                        int valueLowerBound,
                                                                                        int valueHigherBound,
                                                                                        int randomDiscrepanciesAmount)

        {
            var data = new List<DateData>(howMany);

            for (int i = 0; i < howMany; i++)
            {
                data.Add(new DateData(startingDate.AddDays(7 * i), new Random().Next(valueLowerBound, valueHigherBound)));
            }

            //discrepancies adding :
            for (int i = 0; i < randomDiscrepanciesAmount; i++)
            {
                var index = new Random().Next(0, howMany - 1);
                data[index] = (new DateData(data[index].Date, data[index].Value * 8)); //howBigDiscrepancyIs - lets say fixed 8 times more
            }
            return data;
        }

        public IEnumerable<DateData> GenerateLinearDataSetWithDateIndexedWeekly(int howMany,
                                                                                DateTime startingDate,
                                                                                float linearDiscrepancy)
        {
            var data = new List<DateData>(howMany);

            for (int i = 0; i < howMany; i++)
            {
                if (i > 10 && i % 8 == 0 && linearDiscrepancy != 0)
                {
                    //some periodic discrepancies added (modulo 8)
                    data.Add(new DateData(startingDate.AddDays(7 * i), new Random().Next((int)(i - i * linearDiscrepancy), (int)(i + i * linearDiscrepancy))));
                }
                else
                {
                    data.Add(new DateData(startingDate.AddDays(7 * i), i));
                }
            }

            if (linearDiscrepancy != 0)
            {
                //lets add even bigger 5 discrepancies :
                for (int i = 0; i < 5; i++)
                {
                    var index = new Random().Next(0, howMany - 1);
                    data[index] = (new DateData(data[index].Date, data[index].Value * 5)); //howBigDiscrepancyIs - lets say fixed 5 times
                }
            }
            return data;
        }

        public IEnumerable<DateData> GenerateSinusDataSetWithDateIndexedWeekly(int howMany,
                                                                               DateTime startingDate,
                                                                               int valueHigherBound,
                                                                               float sinusDiscrepancy)
        {
            var data = new List<DateData>(howMany);

            for (int i = 0; i < howMany; i++)
            {
                if (i > 10 && i % 40 == 0)
                {
                    var d = new Random().Next((int)(i - i * sinusDiscrepancy), (int)(i + i * sinusDiscrepancy));
                    data.Add(new DateData(startingDate.AddDays(7 * i), (int)(Math.Sin((d * (Math.PI)) / 180) * valueHigherBound + valueHigherBound)));
                }
                else
                {
                    data.Add(new DateData(startingDate.AddDays(7 * i), (int)(Math.Sin((i * (Math.PI)) / 180) * valueHigherBound) + valueHigherBound));
                }
            }

            return data;
        }
    }
}
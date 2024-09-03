namespace ml_data
{
    public class DateData(DateTime d, int v)
    {
        public DateTime Date { get; } = d;
        public int Value { get; } = v;
        public float DateProjectedToNumberForMl
        {
            get
            {
                //return ((DateTimeOffset)Date).ToUnixTimeSeconds();
                var timeSpan = (Date - new DateTime(2023, 1, 1, 0, 0, 0));
                return (long)timeSpan.TotalDays;
            }
        }
    }
}
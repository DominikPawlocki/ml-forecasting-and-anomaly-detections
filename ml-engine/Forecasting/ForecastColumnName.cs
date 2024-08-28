namespace ml_engine.Forecasting
{
    //Just a property name of the one in the model predicions go against
    public enum ForecastColumnName
    {
        NotUsed = 0,
        Value = 1,
        ValueForMl = 2 //Machine learing likes Single precision not integer !
    }
}

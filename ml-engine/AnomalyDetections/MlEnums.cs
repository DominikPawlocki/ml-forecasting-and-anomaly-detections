using System.Runtime.Serialization;

namespace ml_engine.AnomalyDetections
{
    public enum DetectionMethod
    {
        /// <summary>
        /// Iid - independent identically distributed (i.i.d.) time series based on adaptive kernel density estimations and martingale scores.
        /// </summary>
        [EnumMember(Value = "Iid")]
        Iid,
        /// <summary>
        /// Ssa - predicts spikes in time series using Singular Spectrum Analysis (SSA).
        /// </summary>
        [EnumMember(Value = "Ssa")]
        Ssa
    }

    public enum SrCnnDetectMode
    {
        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag).
        /// </summary>
        [EnumMember(Value = "AnomalyOnly")]
        AnomalyOnly,

        /// <summary>
        ///  In this mode, output (IsAnomaly, AnomalyScore, Mag, ExpectedValue, BoundaryUnit, UpperBoundary, LowerBoundary).
        /// </summary>
        [EnumMember(Value = "AnomalyAndMargin")]
        AnomalyAndMargin,

        /// <summary>
        ///     In this mode, output (IsAnomaly, RawScore, Mag, ExpectedValue).
        /// </summary>
        [EnumMember(Value = "AnomalyAndExpectedValue")]
        AnomalyAndExpectedValue
    }

    public enum SrCnnDeseasonalityMode
    {
        //
        // Summary:
        //     In this mode, the stl decompose algorithm is used to de-seasonality.
        Stl,
        //
        // Summary:
        //     In this mode, the mean value of points in the same position in a period is substracted
        //     to de-seasonality.
        Mean,
        //
        // Summary:
        //     In this mode, the median value of points in the same position in a period is
        //     substracted to de-seasonality.
        Median
    }
}

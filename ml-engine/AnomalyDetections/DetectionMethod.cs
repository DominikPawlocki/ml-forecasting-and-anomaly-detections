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
}

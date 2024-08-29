using ml_data;
using ml_ui.Data;
using System.Reflection;
using ml_ui.Services;
using ml_engine.Forecasting;
using ml_engine.AnomalyDetections;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddAutoMapper(Assembly.GetExecutingAssembly());
builder.Services.AddSingleton<WeatherForecastService>();
builder.Services.AddSingleton<IDataGenerator, DataGenerator>();
builder.Services.AddSingleton<WebSocketDataConnector>();
builder.Services.AddSingleton<IMlForecaster, Forecaster>();
builder.Services.AddSingleton<IMlForecastingService, MlForecastingService>();
//builder.Services.AddSingleton(typeof(MLContext));
builder.Services.AddSingleton<ISpikesDetector, SpikesDetector>();
builder.Services.AddSingleton<IAnomalyDetector, AnomalyDetector>();
builder.Services.AddSingleton<IChangePointsDetector, ChangePointsDetector>();
builder.Services.AddSingleton<IMlDataAnomaliesDetectingService, MlDataAnomaliesDetectingService>();


var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();

app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

app.Run();

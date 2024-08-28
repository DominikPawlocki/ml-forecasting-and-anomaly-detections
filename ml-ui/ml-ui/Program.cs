using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using ml_data;
using ml_ui.Data;
using ml_engine;
using System.Reflection;
using ml_ui.Services;

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

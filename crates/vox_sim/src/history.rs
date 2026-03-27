use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A data point in a time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub time: f64,
    pub value: f64,
}

/// A time series for graphing historical data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub name: String,
    pub data: VecDeque<DataPoint>,
    pub max_points: usize,
}

impl TimeSeries {
    pub fn new(name: &str, max_points: usize) -> Self {
        Self {
            name: name.to_string(),
            data: VecDeque::new(),
            max_points,
        }
    }

    pub fn record(&mut self, time: f64, value: f64) {
        self.data.push_back(DataPoint { time, value });
        while self.data.len() > self.max_points {
            self.data.pop_front();
        }
    }

    pub fn latest(&self) -> Option<f64> {
        self.data.back().map(|d| d.value)
    }

    pub fn min(&self) -> f64 {
        self.data.iter().map(|d| d.value).fold(f64::MAX, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.data.iter().map(|d| d.value).fold(f64::MIN, f64::max)
    }

    pub fn average(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|d| d.value).sum::<f64>() / self.data.len() as f64
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Tracks all historical game metrics.
pub struct GameHistory {
    pub population: TimeSeries,
    pub funds: TimeSeries,
    pub income: TimeSeries,
    pub expenses: TimeSeries,
    pub satisfaction: TimeSeries,
    pub crime_rate: TimeSeries,
    pub pollution_avg: TimeSeries,
    pub traffic_density: TimeSeries,
}

impl GameHistory {
    pub fn new(max_points: usize) -> Self {
        Self {
            population: TimeSeries::new("Population", max_points),
            funds: TimeSeries::new("Funds", max_points),
            income: TimeSeries::new("Income", max_points),
            expenses: TimeSeries::new("Expenses", max_points),
            satisfaction: TimeSeries::new("Satisfaction", max_points),
            crime_rate: TimeSeries::new("Crime Rate", max_points),
            pollution_avg: TimeSeries::new("Pollution", max_points),
            traffic_density: TimeSeries::new("Traffic", max_points),
        }
    }

    /// Record current game state as a data point.
    pub fn record_tick(
        &mut self,
        time: f64,
        population: u32,
        funds: f64,
        income: f64,
        expenses: f64,
        satisfaction: f32,
        crime: f32,
        pollution: f32,
        traffic: f32,
    ) {
        self.population.record(time, population as f64);
        self.funds.record(time, funds);
        self.income.record(time, income);
        self.expenses.record(time, expenses);
        self.satisfaction.record(time, satisfaction as f64);
        self.crime_rate.record(time, crime as f64);
        self.pollution_avg.record(time, pollution as f64);
        self.traffic_density.record(time, traffic as f64);
    }
}

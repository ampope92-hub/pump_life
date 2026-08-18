#!/usr/bin/env bash
set -euo pipefail

# Reference solution for pump_life.
#
# The catch is what runtime_days means. For a pump with still_running = 1 the
# seal had NOT failed yet, so that number is a lower bound on its life, not the
# life. Nearly 38% of the export is in that state. Fit on the changed-out pumps
# only and you are fitting the pumps that died young, which lands around R2 0.28
# and predicts an average life well under what the quote batch actually managed.
#
# Worse, the install waves were not random: the long-lived duty types and the
# gentler fluids came online late, so they have short windows and are the most
# likely to still be running. Throwing the censored rows away therefore bends the
# coefficients, not just the intercept. It under-credits exactly the setups that
# last longest.
#
# So: log-normal accelerated failure time, fitted by EM over every row. Censored
# rows enter each pass at the expectation of a normal truncated at where they had
# got to, E[y | y > c] = mu + sd * phi(a)/(1 - Phi(a)), which pulls them up to
# where that pump would plausibly have ended rather than pinning them at the
# report date. Ten passes is plenty. Scores ~0.49 against a noise ceiling of 0.50.

cat > /app/predictor/src/main.rs <<'RUST'
use std::error::Error;
use std::io::Write;

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;

const DUTIES: [&str; 3] = ["continuous", "intermittent", "standby"];
const FLUIDS: [&str; 4] = ["brine", "slurry", "solvent", "water"];

struct Pump {
    pump_id: String,
    duty_cycle: String,
    fluid: String,
    inlet_bar: f64,
    rpm: f64,
    temp_c: f64,
    vibration_mm_s: f64,
    impeller_mm: f64,
    runtime_days: f64,
    still_running: bool,
}

fn read(path: &str) -> Result<Vec<Pump>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let at = |name: &str| headers.iter().position(|h| h == name);
    let i_id = at("pump_id");
    let i_run = at("runtime_days");
    let i_still = at("still_running");
    let col = |name: &str| at(name).unwrap();
    let i_duty = col("duty_cycle");
    let i_fluid = col("fluid");
    let i_inlet = col("inlet_bar");
    let i_rpm = col("rpm");
    let i_temp = col("temp_c");
    let i_vib = col("vibration_mm_s");
    let i_imp = col("impeller_mm");

    let mut out = Vec::new();
    for rec in rdr.records() {
        let r = rec?;
        let f = |i: usize| r[i].parse::<f64>().unwrap();
        out.push(Pump {
            pump_id: i_id.map(|i| r[i].to_string()).unwrap_or_default(),
            duty_cycle: r[i_duty].to_string(),
            fluid: r[i_fluid].to_string(),
            inlet_bar: f(i_inlet),
            rpm: f(i_rpm),
            temp_c: f(i_temp),
            vibration_mm_s: f(i_vib),
            impeller_mm: f(i_imp),
            runtime_days: i_run.map(f).unwrap_or(0.0),
            still_running: i_still.map(|i| f(i) > 0.5).unwrap_or(false),
        });
    }
    Ok(out)
}

fn design(pumps: &[Pump]) -> Vec<Vec<f64>> {
    pumps
        .iter()
        .map(|p| {
            let mut v = vec![p.inlet_bar, p.rpm, p.temp_c, p.vibration_mm_s, p.impeller_mm];
            v.extend(DUTIES.iter().map(|d| if *d == p.duty_cycle { 1.0 } else { 0.0 }));
            v.extend(FLUIDS.iter().map(|f| if *f == p.fluid { 1.0 } else { 0.0 }));
            v
        })
        .collect()
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF, Abramowitz & Stegun 26.2.17 (no erf in std).
fn norm_cdf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let upper = norm_pdf(z) * poly;
    0.5 + sign * (0.5 - upper)
}

fn main() -> Result<(), Box<dyn Error>> {
    let train = read("/app/data/train.csv")?;
    let quote = read("/app/data/test_features.csv")?;

    let xtr_v = design(&train);
    let xtr: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&xtr_v);
    let xte: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&design(&quote));

    // observed log-runtime; for a still-running pump this is only a lower bound
    let obs: Vec<f64> = train.iter().map(|p| p.runtime_days.ln()).collect();
    let censored: Vec<bool> = train.iter().map(|p| p.still_running).collect();

    let mut y = obs.clone();
    let mut model = LinearRegression::fit(&xtr, &y, Default::default())?;
    for _ in 0..10 {
        let mu = model.predict(&xtr)?;
        // residual scale from the pumps that actually failed
        let mut ss = 0.0;
        let mut n = 0.0;
        for i in 0..y.len() {
            if !censored[i] {
                ss += (obs[i] - mu[i]).powi(2);
                n += 1.0;
            }
        }
        let sd = (ss / n).sqrt().max(1e-3);
        for i in 0..y.len() {
            if censored[i] {
                let a = (obs[i] - mu[i]) / sd;
                let lambda = norm_pdf(a) / (1.0 - norm_cdf(a)).max(1e-9);
                y[i] = mu[i] + sd * lambda;
            }
        }
        model = LinearRegression::fit(&xtr, &y, Default::default())?;
    }

    let pred = model.predict(&xte)?;
    let mut out = std::fs::File::create("/app/predictions.csv")?;
    writeln!(out, "pump_id,life_days")?;
    for (p, mu) in quote.iter().zip(&pred) {
        writeln!(out, "{},{:.2}", p.pump_id, mu.exp())?;
    }
    println!("wrote /app/predictions.csv ({} pumps)", quote.len());
    Ok(())
}
RUST

cd /app/predictor
cargo run --release --offline

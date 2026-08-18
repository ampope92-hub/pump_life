use std::error::Error;
use std::io::Write;

use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;

const DUTIES: [&str; 3] = ["continuous", "intermittent", "standby"];
const FLUIDS: [&str; 4] = ["brine", "slurry", "solvent", "water"];

// Phi^-1(0.10). The service interval is the age by which a tenth of the pumps in a given
// configuration have failed, so it is the 0.10 quantile of that pump's own life distribution.
const Z_Q: f64 = -1.281_551_565_544_600_4;

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
    censored: bool,
}

fn read(path: &str) -> Result<Vec<Pump>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let at = |name: &str| headers.iter().position(|h| h == name);
    let i_id = at("pump_id");
    let i_run = at("runtime_days");
    let i_win = at("observed_window_days");
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
        let runtime = i_run.map(f).unwrap_or(0.0);
        // No flag is shipped. A pump whose seal has been changed out has a runtime shorter
        // than the window it has been installed for; one still in service has run for the
        // whole window, so the two columns are equal and its runtime is a lower bound.
        let censored = match i_win {
            Some(i) => (runtime - f(i)).abs() < 1e-9,
            None => false,
        };
        out.push(Pump {
            pump_id: i_id.map(|i| r[i].to_string()).unwrap_or_default(),
            duty_cycle: r[i_duty].to_string(),
            fluid: r[i_fluid].to_string(),
            inlet_bar: f(i_inlet),
            rpm: f(i_rpm),
            temp_c: f(i_temp),
            vibration_mm_s: f(i_vib),
            impeller_mm: f(i_imp),
            runtime_days: runtime,
            censored,
        });
    }
    Ok(out)
}

/// Mean model: intercept, duty and fluid one-hots (first level dropped), centred numerics.
fn design(pumps: &[Pump]) -> Vec<Vec<f64>> {
    pumps
        .iter()
        .map(|p| {
            let mut row = vec![1.0];
            for d in &DUTIES[1..] {
                row.push(if p.duty_cycle == *d { 1.0 } else { 0.0 });
            }
            for f in &FLUIDS[1..] {
                row.push(if p.fluid == *f { 1.0 } else { 0.0 });
            }
            row.push(p.inlet_bar - 4.2);
            row.push(p.rpm - 2900.0);
            row.push(p.temp_c - 52.0);
            row.push(p.vibration_mm_s - 3.1);
            row.push(p.impeller_mm - 145.0);
            row
        })
        .collect()
}

/// Scale model: log sigma is linear in vibration. Seal life is not merely shorter on a
/// rough pump, it is less predictable, and the interval depends on that spread.
fn scale_design(pumps: &[Pump]) -> Vec<Vec<f64>> {
    pumps
        .iter()
        .map(|p| vec![1.0, p.vibration_mm_s - 3.1])
        .collect()
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Abramowitz & Stegun 26.2.17 — Rust's standard library has no error function.
fn norm_cdf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let z = x.abs() / 2.0_f64.sqrt();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-z * z).exp();
    0.5 * (1.0 + sign * erf)
}

/// log(1 - Phi(z)), guarded so the far upper tail cannot underflow to -inf.
fn log_sf(z: f64) -> f64 {
    let sf = norm_cdf(-z);
    if sf < 1e-300 {
        // asymptotic: log(phi(z)/z) for large z
        norm_pdf(z).ln() - z.ln()
    } else {
        sf.ln()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Exact log-likelihood of a right-censored log-normal with a covariate-dependent scale.
///
/// Failed pumps contribute their density; still-running pumps contribute the probability
/// of surviving past where they have got to. Dropping the second group is what biases the
/// fit, and it biases the spread harder than the level: the residuals you can still see
/// are the ones that failed early.
fn loglik(x: &[Vec<f64>], z: &[Vec<f64>], y: &[f64], cens: &[bool], b: &[f64], g: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..y.len() {
        let mu = dot(&x[i], b);
        let log_sd = dot(&z[i], g).clamp(-3.0, 1.5);
        let sd = log_sd.exp();
        let t = (y[i] - mu) / sd;
        total += if cens[i] {
            log_sf(t)
        } else {
            -0.5 * t * t - log_sd
        };
    }
    total
}

/// Analytic gradient of `loglik` in (beta, gamma).
fn grad(
    x: &[Vec<f64>],
    z: &[Vec<f64>],
    y: &[f64],
    cens: &[bool],
    b: &[f64],
    g: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut gb = vec![0.0; b.len()];
    let mut gg = vec![0.0; g.len()];
    for i in 0..y.len() {
        let mu = dot(&x[i], b);
        let log_sd = dot(&z[i], g).clamp(-3.0, 1.5);
        let sd = log_sd.exp();
        let t = (y[i] - mu) / sd;
        // d/dmu and d/dlog_sd of this row's contribution
        let (dmu, dls) = if cens[i] {
            // inverse Mills ratio; the survival term's derivatives
            let lambda = norm_pdf(t) / norm_cdf(-t).max(1e-12);
            (lambda / sd, lambda * t)
        } else {
            (t / sd, t * t - 1.0)
        };
        for (j, xij) in x[i].iter().enumerate() {
            gb[j] += dmu * xij;
        }
        for (k, zik) in z[i].iter().enumerate() {
            gg[k] += dls * zik;
        }
    }
    (gb, gg)
}

fn main() -> Result<(), Box<dyn Error>> {
    let train = read("/app/data/train.csv")?;
    let quote = read("/app/data/test_features.csv")?;

    let xtr = design(&train);
    let ztr = scale_design(&train);
    let y: Vec<f64> = train.iter().map(|p| p.runtime_days.ln()).collect();
    let cens: Vec<bool> = train.iter().map(|p| p.censored).collect();
    let n_failed = cens.iter().filter(|c| !**c).count();
    println!(
        "{} pumps, {} still running ({:.1}% censored)",
        train.len(),
        train.len() - n_failed,
        100.0 * (train.len() - n_failed) as f64 / train.len() as f64
    );

    // Columns differ in scale by three orders of magnitude (one-hots against centred rpm),
    // so a single step size cannot serve them all. Standardise for the optimisation and
    // undo it at the end.
    let ncol = xtr[0].len();
    let mut xsd = vec![1.0; ncol];
    for j in 1..ncol {
        let m: f64 = xtr.iter().map(|r| r[j]).sum::<f64>() / xtr.len() as f64;
        let v: f64 = xtr.iter().map(|r| (r[j] - m).powi(2)).sum::<f64>() / xtr.len() as f64;
        xsd[j] = v.sqrt().max(1e-12);
    }
    let scaled = |rows: &[Vec<f64>]| -> Vec<Vec<f64>> {
        rows.iter()
            .map(|r| r.iter().zip(&xsd).map(|(v, s)| v / s).collect())
            .collect()
    };
    let xs = scaled(&xtr);

    // Start from a complete-case fit: biased, but in the right region for the optimiser.
    let obs: Vec<usize> = (0..train.len()).filter(|&i| !cens[i]).collect();
    let feats: Vec<Vec<f64>> = obs.iter().map(|&i| xs[i][1..].to_vec()).collect();
    let targ: Vec<f64> = obs.iter().map(|&i| y[i]).collect();
    let m0 = LinearRegression::fit(&DenseMatrix::from_2d_vec(&feats), &targ, Default::default())?;
    let mut beta = vec![0.0; ncol];
    beta[0] = *m0.intercept();
    for (j, c) in m0.coefficients().iterator(0).enumerate() {
        beta[j + 1] = *c;
    }
    let resid_sd = {
        let fitted = m0.predict(&DenseMatrix::from_2d_vec(&feats))?;
        let v: f64 = targ
            .iter()
            .zip(&fitted)
            .map(|(a, f)| (a - f).powi(2))
            .sum::<f64>()
            / targ.len() as f64;
        v.sqrt()
    };
    let mut gamma = vec![resid_sd.ln(), 0.0];

    // Gradient ascent with a backtracking step. The likelihood is smooth and concave in
    // practice here, so this converges without needing a Hessian.
    let mut step = 1e-3;
    let mut ll = loglik(&xs, &ztr, &y, &cens, &beta, &gamma);
    for _ in 0..4000 {
        let (gb, gg) = grad(&xs, &ztr, &y, &cens, &beta, &gamma);
        let norm = (gb.iter().chain(&gg).map(|v| v * v).sum::<f64>()).sqrt();
        if norm < 1e-7 {
            break;
        }
        let mut accepted = false;
        for _ in 0..40 {
            let nb: Vec<f64> = beta.iter().zip(&gb).map(|(v, d)| v + step * d).collect();
            let ng: Vec<f64> = gamma.iter().zip(&gg).map(|(v, d)| v + step * d).collect();
            let nl = loglik(&xs, &ztr, &y, &cens, &nb, &ng);
            if nl > ll {
                beta = nb;
                gamma = ng;
                ll = nl;
                step *= 1.3;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    println!(
        "censored MLE: loglik {:.2}, log-sigma = {:.4} {:+.4} * (vibration - 3.1)",
        ll, gamma[0], gamma[1]
    );

    // Undo the standardisation so the coefficients apply to the raw design.
    let beta_raw: Vec<f64> = beta.iter().zip(&xsd).map(|(b, s)| b / s).collect();

    let xq = design(&quote);
    let zq = scale_design(&quote);
    let mut out = std::fs::File::create("/app/predictions.csv")?;
    writeln!(out, "pump_id,life_days,service_days")?;
    let mut life_sum = 0.0;
    let mut serv_sum = 0.0;
    for i in 0..quote.len() {
        let mu = dot(&xq[i], &beta_raw);
        let sd = dot(&zq[i], &gamma).clamp(-3.0, 1.5).exp();
        // A log-normal's mean sits above its median by sigma^2/2; the interval is its
        // lower quantile. Both move with sigma, in opposite directions.
        let life = (mu + 0.5 * sd * sd).exp();
        let service = (mu + sd * Z_Q).exp();
        life_sum += life;
        serv_sum += service;
        writeln!(out, "{},{:.2},{:.2}", quote[i].pump_id, life, service)?;
    }
    println!(
        "wrote /app/predictions.csv ({} pumps, mean life {:.1}, mean service {:.1})",
        quote.len(),
        life_sum / quote.len() as f64,
        serv_sum / quote.len() as f64
    );
    Ok(())
}

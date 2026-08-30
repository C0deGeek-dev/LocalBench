//! Statistics primitives shared by the scoring and uplift math.

/// Median of the values; `0.0` for an empty slice.
#[must_use]
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Arithmetic mean of the values; `0.0` for an empty slice.
#[must_use]
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation (n-1 denominator); `0.0` with fewer than two
/// values.
#[must_use]
pub fn sample_std_dev(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(values);
    let sum_sq: f64 = values.iter().map(|v| (v - m) * (v - m)).sum();
    (sum_sq / (n as f64 - 1.0)).sqrt()
}

/// Round to `dp` decimal places (half away from zero).
#[must_use]
pub fn round_dp(value: f64, dp: u32) -> f64 {
    let scale = 10f64.powi(dp as i32);
    (value * scale).round() / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_odd_even_and_empty() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[3.0]), 3.0);
        assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn mean_and_stddev() {
        assert_eq!(mean(&[]), 0.0);
        assert!((mean(&[2.0, 4.0, 6.0]) - 4.0).abs() < f64::EPSILON);
        assert_eq!(sample_std_dev(&[5.0]), 0.0);
        let sd = sample_std_dev(&[2.0, 4.0, 6.0]);
        assert!(
            (sd - 2.0).abs() < 1e-12,
            "sample stddev of {{2,4,6}} is 2.0"
        );
    }

    #[test]
    fn rounding_pins_decimal_places() {
        assert_eq!(round_dp(396.666_666, 2), 396.67);
        assert_eq!(round_dp(0.777_78, 4), 0.7778);
    }
}

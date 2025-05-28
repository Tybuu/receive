use std::{num, vec};

use ndarray::{Array, Array1, ArrayBase, Axis, Zip, array, concatenate, linspace, stack};
use plotly::{Layout, Plot, Scatter};
// use rtlsdr::RTLSDRDevice;
use rustfft::{
    FftPlanner,
    num_complex::{Complex, Complex64},
};

// RTL Settings
const FSPS: u64 = 2 * 256 * 256 * 16; // about 2Msps...works
const FC: f64 = 250e6;
const TMAX: f64 = 122.076e6 * 80.0;
const NYQ: f64 = (FSPS / 2) as f64;

// Backscatter Settings
const FREQ_LOW: f64 = 50e3;
const FREQ_HIGH: f64 = 100e3;
const BIT_TIME: f64 = 500e-6;

const BORDER: f64 = FREQ_LOW + (FREQ_HIGH - FREQ_LOW) / 2.0;

const SAMPLES_PER_BIT: u64 = (FSPS as f64 * BIT_TIME) as u64;
const NUM_DATA_BITS: u64 = 6;

fn main() {
    let n = (FSPS as f64 * BIT_TIME).round() as usize;
    println!("{}", n * 2);
    let high_wave = Array::from_elem((FSPS as f64 * BIT_TIME) as usize, FREQ_HIGH);
    let low_wave = Array::from_elem((FSPS as f64 * BIT_TIME) as usize, FREQ_LOW);

    let ref_freq = stack![
        Axis(0),
        high_wave,
        high_wave,
        high_wave,
        low_wave,
        low_wave,
        high_wave,
        low_wave
    ]
    .flatten()
    .into_owned();

    let bandpass = bandpass(n);

    // let trace = Scatter::new(
    //     Array1::linspace(-NYQ, NYQ, n).into_raw_vec_and_offset().0,
    //     bandpass.clone().into_raw_vec_and_offset().0,
    // );
    // let mut plt = Plot::new();
    // plt.add_trace(trace);
    //
    // let layout = Layout::new().x_axis(plotly::layout::Axis::new().range(vec![-200e3, 200e3]));
    // plt.set_layout(layout);
    // plt.show();
    //
    // let mut samples: Array1<Complex64> = Array1::zeros(n);
    let mut prev_buffer: Array1<Complex64> =
        Array1::zeros(ref_freq.len() + (SAMPLES_PER_BIT * NUM_DATA_BITS) as usize);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let mut fft_scratch = vec![Complex64::ZERO; fft.get_inplace_scratch_len()];
    let mut ifft_scratch = vec![Complex64::ZERO; ifft.get_inplace_scratch_len()];
    loop {
        let samples = read_samples(n);
        let mut buffer = stack![Axis(0), samples, prev_buffer]
            .flatten()
            .into_owned()
            .into_raw_vec_and_offset()
            .0;
        fft.process_with_scratch(&mut buffer, &mut fft_scratch);
    }
}

fn bandpass(num: usize) -> Array1<f64> {
    let mut freqs = Array::linspace(-NYQ, NYQ, num);
    Zip::from(&mut freqs).for_each(|x| {
        if *x >= FREQ_LOW - 20e3 && *x <= FREQ_HIGH + 20e3 {
            *x = 1.0;
        } else {
            *x = 0.0;
        }
    });
    freqs
}

#[inline]
fn ncc(va: &Array1<f64>, vb: &Array1<f64>) -> f64 {
    return (va.dot(vb) / (va.dot(va) * vb.dot(vb)).sqrt()).abs();
}

fn read_samples(n: usize /*sdr: &mut RTLSDRDevice */) -> Array1<Complex64> {
    let buf: Array1<f64> = Array1::linspace(0., (2 * n) as f64, 2 * n);
    // let buf = match sdr.read_sync(n * 2) {
    //     Ok(val) => Array1::from(val),
    //     Err(e) => panic!("{}", e),
    // };
    buf.exact_chunks(2)
        .into_iter()
        .map(|x| Complex64::new(x[0], x[1]))
        .collect()
}

fn process(data: &Array1<Complex64>) {}

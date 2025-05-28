use ndarray::{Array, Array1, ArrayBase, Axis, Zip, array, concatenate, stack};
use plotly::{Layout, Plot, Scatter};

// RTL Settings
const FSPS: u64 = 2 * 256 * 256 * 16; // about 2Msps...works
const FC: f64 = 250e6;
const TMAX: f64 = 122.076e6 * 80.0;
const NYQ: f64 = (FSPS / 2) as f64;

// Backscatter Settings
const FREQ_LOW: f64 = 50e3;
const FREQ_HIGH: f64 = 100e3;
const BIT_TIME: f64 = 500e-6;

fn main() {
    let n = (FSPS as f64 * BIT_TIME).round() as usize;
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

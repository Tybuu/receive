use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::VecDeque,
    f64::consts::PI,
    num,
    sync::{Mutex, mpsc},
    time::Instant,
    vec,
};

use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayView1, AssignElem, Axis, Zip, array, azip, concatenate,
    linspace, s, stack,
};
use once_cell::sync::Lazy;
use plotly::{Layout, Plot, Scatter};
// use rtlsdr::RTLSDRDevice;
use rayon::prelude::*;
use rtlsdr::RTLSDRDevice;
use rustfft::{
    FftPlanner,
    num_complex::{Complex, Complex64, ComplexFloat},
    num_traits::{Euclid, Zero},
};
use uinput::event::keyboard::Key;

// RTL Settings
const FSPS: u64 = 2 * 256 * 256 * 16; // about 2Msps...works
const FC: u32 = 250_000_000;
const TMAX: f64 = BIT_TIME * 26.0;
const NYQ: f64 = (FSPS / 2) as f64;

// Backscatter Settings
const FREQ_LOW: f64 = 300e3;
const FREQ_HIGH: f64 = 600e3;
const BIT_TIME: f64 = 25e-6;

const BORDER: f64 = FREQ_LOW + (FREQ_HIGH - FREQ_LOW) / 2.0;

const SAMPLES_PER_BIT: u64 = (FSPS as f64 * BIT_TIME) as u64;
const NUM_DATA_BITS: u64 = 6;
const REF_FREQ: Lazy<Array1<f64>> = Lazy::new(|| {
    let high_wave = Array::from_elem((FSPS as f64 * BIT_TIME) as usize, 1.0);
    let low_wave = Array::from_elem((FSPS as f64 * BIT_TIME) as usize, -1.0);
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
    ref_freq
});

fn main() {
    //
    let n = (FSPS as f64 * TMAX).floor() as usize;
    let n = (n as usize / 256) * 256;
    let (samples_tx, samples_rx) = mpsc::channel();

    let rtl_handle = std::thread::spawn(move || {
        let mut sdr = rtlsdr::open(0).unwrap();
        if n % 256 != 0 {
            panic!(
                "N needs to be a multiple of 256. {} is not a multiple of 256",
                n
            );
        }
        println!("Configuring for {} samples", n);
        sdr.set_sample_rate(FSPS as u32)
            .expect("Invalid Sample Rate");
        sdr.set_center_freq(FC).expect("Invalid Center Frequency");
        sdr.set_agc_mode(false).expect("Failed to disable agc mode");
        sdr.set_tuner_gain_mode(true)
            .expect("Failed to set manual gain");
        println!("Sdr Gain Values: {:?}", sdr.get_tuner_gains().unwrap());
        sdr.set_tuner_gain(496).expect("Invalid Gain"); // Appoximately 20 db
        sdr.reset_buffer().expect("Failed to reset buffer");
        loop {
            let data = read_samples(n, &mut sdr);
            samples_tx.send(data).unwrap();
        }
    });

    let process_handle = std::thread::spawn(move || {
        let mut prev_buffer: Array1<Complex64> =
            Array1::zeros(REF_FREQ.len() + (SAMPLES_PER_BIT * NUM_DATA_BITS) as usize);

        let buffer_size = prev_buffer.len() + n;
        let fft_size = prev_buffer.len() + n;
        let mut buffer: Array1<Complex64> = Array1::zeros(fft_size);
        let mut abs_buffer: Array1<f64> = Array1::zeros(buffer_size);
        let mut theta_clone: Array1<f64> = Array1::zeros(buffer_size);
        let mut theta_x: Array1<f64> = Array1::zeros(buffer_size);
        let mut theta_buffer: Array1<f64> = Array1::zeros(buffer_size);
        println!(
            "n: {}, prev: {}, total: {}",
            n,
            prev_buffer.len(),
            buffer_size
        );

        let prev_buffer_size = prev_buffer.len();
        let bandpass = bandpass(fft_size);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);
        let mut fft_scratch = vec![Complex64::ZERO; fft.get_inplace_scratch_len()];
        let mut ifft_scratch = vec![Complex64::ZERO; ifft.get_inplace_scratch_len()];

        let mut overlap = false;
        let mut plot = false;
        let mut count = 0;
        let mut correct_bits = Array1::from(vec![0, 0, 0, 1, 1, 1]);
        loop {
            // Read the samples and combine it with the previous buffer
            let samples = samples_rx.recv().unwrap();
            let start = Instant::now();
            buffer
                .slice_mut(s![0..prev_buffer.len()])
                .assign(&prev_buffer);
            buffer
                .slice_mut(s![prev_buffer.len()..buffer_size])
                .assign(&samples);
            buffer
                .slice_mut(s![buffer_size..])
                .mapv_inplace(|_| Complex64::zero());

            // Apply fft to buffer
            fft.process_with_scratch(buffer.as_slice_mut().unwrap(), &mut fft_scratch);

            // Elementwise multiplication in place
            Zip::from(&mut buffer)
                .and(&bandpass)
                .for_each(|x, y| *x *= y);

            // Revert back into time domain
            ifft.process_with_scratch(buffer.as_slice_mut().unwrap(), &mut ifft_scratch);
            buffer.mapv_inplace(|x| x / buffer_size as f64);
            let fft_time = Instant::now();

            let mut data_buffer = buffer.slice_mut(s![..buffer_size]);
            // Squelch our data
            Zip::from(&mut abs_buffer)
                .and(&data_buffer)
                .for_each(|x, y| *x = y.re * y.re + y.im * y.im);
            let mean = abs_buffer.mean().unwrap_or_default();
            let threshold = mean / 9.0;

            Zip::from(&mut data_buffer)
                .and(&abs_buffer)
                .for_each(|x, y| {
                    if *y < threshold {
                        *x = Complex64::zero();
                    }
                });

            // Find the derivative
            Zip::from(theta_buffer.slice_mut(s![1..]))
                .and(data_buffer.slice(s![..-1]))
                .and(data_buffer.slice(s![1..]))
                .par_for_each(|x, y, z| {
                    let left_theta = y.arg();
                    let right_theta = z.arg();
                    let deriv = right_theta - left_theta;

                    let left_theta_pi = (left_theta + PI).rem_euclid(2.0 * PI);
                    let right_theta_pi = (right_theta + PI).rem_euclid(2.0 * PI);
                    let deriv_pi = right_theta_pi - left_theta_pi;

                    *x = if deriv.abs() < deriv_pi.abs() {
                        deriv
                    } else {
                        deriv_pi
                    }
                });

            // Apply the spike threshold
            let spikethresh = 2.0;
            theta_clone.assign(&theta_buffer);
            Zip::from(theta_buffer.slice_mut(s![1..-1]))
                .and(&theta_clone.slice(s![0..-2]))
                .and(&theta_clone.slice(s![2..]))
                .par_for_each(|x, y, z| {
                    if x.abs() > spikethresh {
                        *x = (*z + *y) / 2.0;
                    }
                });

            // Threshold frequencies into bits
            theta_buffer.mapv_inplace(|x| {
                let res = x * FSPS as f64 / (2.0 * PI);
                if res > BORDER { 1.0 } else { -1.0 }
            });
            let filter_time = Instant::now();

            // Slide a window through frequencies and find the highest correlation score
            let res = (0..n)
                .into_par_iter()
                // .step_by(SAMPLES_PER_BIT as usize / 16)
                .filter(|&i| !(i < prev_buffer_size && overlap))
                .map(|i| {
                    let samples = theta_buffer.slice(s![i..i + REF_FREQ.len()]);
                    let score = ncc(&samples, &REF_FREQ.view());

                    // println!("Score: {}", score);
                    (i, score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .unwrap();
            let ncc_time = Instant::now();

            // Process the packet if the socre is higher than the threshold
            if res.1 > 0.80 {
                // if res.1 < 0.80 && !plot {
                //     let trace = Scatter::new(
                //         (0..(SAMPLES_PER_BIT as usize * REF_FREQ.len())).collect::<Vec<_>>(),
                //         theta_buffer
                //             .slice(s![res.0..(res.0 + REF_FREQ.len())])
                //             .to_vec(),
                //     );
                //     let trace2 = Scatter::new(
                //         (0..(SAMPLES_PER_BIT as usize * REF_FREQ.len())).collect::<Vec<_>>(),
                //         REF_FREQ.to_vec(),
                //     );
                //     let mut plot_graph = Plot::new();
                //     plot_graph.add_trace(trace);
                //     plot_graph.add_trace(trace2);
                //     plot_graph.show();
                //     plot = true;
                // }
                // If our last bit is within the next samples previous buffer, we want to skip it
                // as it could influence the correlation score. We won't miss any packets either as
                // the backscatter device has a packet time worth delay between each packet
                overlap = res.0 > buffer_size - 2 * prev_buffer_size;
                // Seperates the samples into their own bit buckets
                let bit_start = res.0 + REF_FREQ.len();
                let bit_end = bit_start + (NUM_DATA_BITS * SAMPLES_PER_BIT) as usize;
                let bits = theta_buffer
                    .slice(s![bit_start..bit_end])
                    .into_shape_with_order((NUM_DATA_BITS as usize, SAMPLES_PER_BIT as usize))
                    .unwrap();
                let mut bits = bits.sum_axis(Axis(1));
                let bits = bits.mapv(|x| if x as u64 > 0 { 1 } else { 0 });
                correct_bits.mapv_inplace(|x| x ^ 1);
                count += 1;

                if !bits.eq(&correct_bits) {
                    println!("Failed at count: {}", count);
                    println!("Correct Bits: {}", correct_bits);
                    println!("Actual Bits: {}", bits);
                    println!("Score: {}", res.1,);
                    println!("Pos: {}", res.0);
                    std::process::exit(0);
                }
                println!(
                    "Count: {} | Data: {} | Score: {} | Position: {} | Buffer Len: {} | Prev buffer Len: {}",
                    count, bits, res.1, res.0, buffer_size, prev_buffer_size
                );
            } else {
                overlap = false;
            }
            prev_buffer.assign(&samples.slice(s![(samples.len() - prev_buffer_size)..]));
        }
    });

    rtl_handle.join().unwrap();
    process_handle.join().unwrap();
}

fn bandpass(num: usize) -> Array1<Complex64> {
    let half = num / 2;
    let first_half = Array::linspace(0.0, NYQ, half + 1);
    let second_half = Array1::zeros(num - half - 1);
    let freqs = concatenate![Axis(0), first_half, second_half]
        .flatten()
        .into_owned();
    freqs
        .into_iter()
        .map(|x| {
            if x >= FREQ_LOW - 10e3 && x <= FREQ_HIGH + 10e3 {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::zero()
            }
        })
        .collect()
}

// #[inline]
// fn ncc(va: &ArrayView1<f64>, vb: &ArrayView1<f64>) -> f64 {
//     let a_m = va.mean().unwrap();
//     let va_m = va.mapv(|x| x - a_m);
//     let b_m = vb.mean().unwrap();
//     let vb_m = vb.mapv(|x| x - b_m);
//     return ((va_m).dot(&vb_m) / (va_m.dot(&va_m) * vb_m.dot(&vb_m)).sqrt()).abs();
// }

#[inline]
fn ncc(va: &ArrayView1<f64>, vb: &ArrayView1<f64>) -> f64 {
    return ((va).dot(vb) / (va.dot(va) * vb.dot(vb)).sqrt()).abs();
}

fn read_samples(n: usize, sdr: &mut RTLSDRDevice) -> Array1<Complex64> {
    // sdr.reset_buffer().expect("Unable to reset buffer");
    // We need to read 2 * n samples as each sample is 2 u8
    let buf = match sdr.read_sync(n * 2) {
        Ok(val) => Array1::from(val),
        Err(e) => panic!("{}", e),
    };
    // sdr devices returns I, Q samples interleaved as u8. We create slices of each pair
    // and center them around [-1, 1]
    buf.exact_chunks(2)
        .into_iter()
        .map(|x| {
            let i = (x[0] as f64 - 127.5) / 128.0;
            let q = (x[1] as f64 - 127.5) / 128.0;
            Complex64::new(i, q)
        })
        .collect()
}

use std::f64::consts::E as e;

fn ensure_finite(x: f64) -> () {
    if !x.is_finite(){
        panic!("Activation is not finite!");
    }
}

pub fn relu(x: f64) -> f64 {
    ensure_finite(x);
    if x < 0.0 {
        0.0
    }
    else {
        x
    }
}

pub fn sigmoid(x: f64) -> f64 {
    ensure_finite(x);
    1.0 / (1.0 + e.powf(-1.0 * x))
}

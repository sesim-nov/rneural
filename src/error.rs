use std::error::Error;

fn error_ms(pred: Vec<f64>, act: Vec<f64>) -> Result<f64, Box<dyn Error>>{
    let m = pred.len();
    if m != act.len() {
        Err("Length of predicted and actual error values must match".into())
    } else {
        let err_sq: f64 = pred.iter().enumerate().map(|x| {
            let (i, pred) = x;
            (pred - act[i]).powi(2)
        }).sum();
        Ok(1.0 / (m as f64) * err_sq)
    }
}

// TODO: Write unit test for this error fn. 

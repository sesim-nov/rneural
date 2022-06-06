// So, i need:
//   * A structure to hold the weight tables
//   * Functions to calculate the activations. (No need to store individual nodes).
//   * Tracking to carry how many layers a network has
//   * Error calculation
//   * Activation functions (mostly done)
//   * Lookup backpropagation stuff.

use ndarray::{array, Array2};
use rneural::neuralnet::{NetRecord, NeuralNet};
use rneural::activations::Relu;
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Deserialize)]
struct HousePrice {
    crim: f64,
    zn: f64,
    indus: f64,
    chas: f64,
    nox: f64,
    rm: f64,
    age: f64,
    dis: f64,
    rad: f64,
    tax: f64,
    ptratio: f64,
    medv: f64,
    b: f64,
    lstat: f64,
}

impl NetRecord for HousePrice {
    fn get_inputs(&self) -> Array2<f64> {
        array![
            [self.crim],
            [self.zn],
            [self.indus],
            [self.chas],
            [self.nox],
            [self.rm],
            [self.age],
            [self.dis],
            [self.rad],
            [self.tax],
            [self.ptratio],
            [self.b],
            [self.lstat],
        ]
    }
    fn get_outputs(&self) -> Array2<f64> {
        array![[self.medv]]
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let path = get_first_arg()?;
    let file = File::open(path)?;
    let mut csvrdr = csv::ReaderBuilder::new().delimiter(b',').from_reader(file);
    let net: NeuralNet::<Relu> = NeuralNet::new_rand(vec![13, 20, 10, 1]);
    for record in csvrdr.deserialize() {
        let record: HousePrice = record?;
        let actual = record.get_outputs();
        let pred_state = net.solve_fwd(record.get_inputs())?;
        println!("Error:\n{}", &pred_state.output - &actual);
        println!("Last Activation:\n{}", pred_state.activations[pred_state.activations.len() - 1]);
        let back = net.back_prop(pred_state, actual)?;
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        panic!("{}", e);
    }
}

fn get_first_arg() -> Result<String, Box<dyn Error>> {
    match env::args().nth(1) {
        None => Err("No file path argument provided.".into()),
        Some(x) => Ok(x),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use rneural::neuralnet::NeuralNet;
    #[test]
    fn net_creation() {
        let net = NeuralNet {
            weights: vec![array![[1.0, 2.0], [3.0, 4.0]]],
            bias: vec![array![[1.0], [1.0]]],
            activation: rneural::activations::relu,
        };
    }
}

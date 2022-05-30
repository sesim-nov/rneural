// So, i need: 
//   * A structure to hold the weight tables
//   * Functions to calculate the activations. (No need to store individual nodes). 
//   * Tracking to carry how many layers a network has
//   * Error calculation
//   * Activation functions (mostly done)
//   * Lookup backpropagation stuff. 

use rneural::neuralnet::NeuralNet;
use ndarray::array;
use std::env;
use std::error::Error;
use std::fs::File;

fn run() -> Result<(), Box<dyn Error>> {
    let path = get_first_arg()?;
    let file = File::open(path)?;
    let mut csvrdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .from_reader(file);
    for record in csvrdr.records() {
        let record = record?;
        println!("{:?}", record);
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        println!("{:?}", e);
        std::process::exit(1);
    }
}

fn get_first_arg() -> Result<String, Box<dyn Error>>{
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
        let net = NeuralNet{
            weights: vec![array![[1.0,2.0],[3.0,4.0]]],
            bias: vec![array![[1.0], [1.0]]],
            activation: rneural::activations::relu,
        };
    }
}

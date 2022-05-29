// So, i need: 
//   * A structure to hold the weight tables
//   * Functions to calculate the activations. (No need to store individual nodes). 
//   * Tracking to carry how many layers a network has
//   * Error calculation
//   * Activation functions (mostly done)
//   * Lookup backpropagation stuff. 

use rneural::neuralnet::NeuralNet;
use ndarray::array;

fn main() {
    rneural::helo();
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

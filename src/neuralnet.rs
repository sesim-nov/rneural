use ndarray::{array, Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;

#[derive(Debug)]
pub struct NeuralNet<T>
where
    T: Fn(f64) -> f64,
{
    pub weights: Vec<Array2<f64>>,
    pub bias: Vec<Array2<f64>>,
    pub activation: T,
}

impl NeuralNet<fn(f64) -> f64> {
    // TODO: Maybe make all fields private and provided specified-weights constructor?
    pub fn new_ones(nodes: Vec<usize>) -> Self {
        let mut weights = Vec::new();
        for shape in nodes.windows(2) {
            weights.push(Array2::<f64>::ones((shape[1], shape[0])));
        }
        let mut bias = Vec::new();
        for shape in &nodes[1..] {
            bias.push(Array2::<f64>::ones((*shape, 1)));
        }
        NeuralNet {
            weights,
            bias,
            activation: crate::activations::relu,
        }
    }
    pub fn new_rand(nodes: Vec<usize>) -> Self {
        let mut weights = Vec::new();
        for shape in nodes.windows(2) {
            weights.push(Array2::<f64>::random(
                (shape[1], shape[0]),
                Uniform::new(0., 3.),
            ));
        }
        let mut bias = Vec::new();
        for shape in &nodes[1..] {
            bias.push(Array2::<f64>::random((*shape, 1), Uniform::new(0., 3.)));
        }
        NeuralNet {
            weights,
            bias,
            activation: crate::activations::relu,
        }
    }
}

impl<T> NeuralNet<T>
where
    T: Fn(f64) -> f64,
{
    // TODO: Appropriately error if a network is invalid (wrong weight matrix shapes, for example).
    /// Process the forward solution of the network given an input vector.
    pub fn solve_fwd(&self, input: Array2<f64>) -> Result<NetState, Box<dyn Error>> {
        let mut all_activations: Vec<Array2<f64>> = Vec::new();
        let mut acts = input.clone();
        let num_weights = self.weights.len();
        for (i, weight) in self.weights.iter().enumerate() {
            acts = {
                let unact = weight.dot(&acts) + &self.bias[i];
                // Avoid applying activation to the output layer.
                if num_weights - 1 == i {
                    unact
                } else {
                    unact.map(|x| (self.activation)(*x))
                }
            };
            all_activations.push(acts.clone());
        }
        all_activations.pop();
        Ok(NetState {
            input,
            activations: all_activations,
            output: acts,
        })
    }
    pub fn back_prop(
        &self,
        input: Array2<f64>,
        output: Array2<f64>,
    ) -> Result<Vec<Array2<f64>>, Box<dyn Error>> {
        Err("Backprop: STUB".into())
    }
}

/// Trait used to parse the inputs and outputs from a CSV record created by the csv crate.
pub trait NetRecord {
    fn get_inputs(&self) -> Array2<f64>;
    fn get_outputs(&self) -> Array2<f64>;
}

/// Tracks the complete neural state of a neural network.
///
/// Returned by the solve_fwd function.
#[derive(Debug)]
pub struct NetState {
    input: Array2<f64>,
    activations: Vec<Array2<f64>>,
    output: Array2<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new() {
        let new = NeuralNet::new_ones(vec![3, 4, 4, 1]);
        assert_eq!(new.weights[0].shape(), &[4, 3]);
        assert_eq!(new.weights[1].shape(), &[4, 4]);
        assert_eq!(new.weights[2].shape(), &[1, 4]);
        assert_eq!(new.bias[0].shape(), &[4, 1]);
        assert_eq!(new.bias[1].shape(), &[4, 1]);
        assert_eq!(new.bias[2].shape(), &[1, 1]);
    }
    #[test]
    fn test_solve() {
        let net = NeuralNet {
            weights: vec![array![[1.0, 1.0], [2.0, 2.0]], array![[1.0, 1.0]]],
            bias: vec![array![[0.0], [0.0]], array![[0.0]]],
            activation: crate::activations::relu,
        };
        let inputs = array![[3.0], [4.0]];
        let soln = net.solve_fwd(inputs);
        assert_eq!(soln, Ok(array![[21.0]]))
    }
}

use crate::activations::{Activation, Relu};
use ndarray::{array, Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;

#[derive(Debug)]
pub struct NeuralNet<T>
where
    T: Activation,
{
    pub weights: Vec<Array2<f64>>,
    pub bias: Vec<Array2<f64>>,
    pub activation: T,
}

impl<T: Activation> NeuralNet<T> {
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
            activation: T::new(),
        }
    }
    pub fn new_rand(nodes: Vec<usize>) -> Self {
        let mut weights = Vec::new();
        for shape in nodes.windows(2) {
            weights.push(Array2::<f64>::random(
                (shape[1], shape[0]),
                Uniform::new(0., 1.),
            ));
        }
        let mut bias = Vec::new();
        for shape in &nodes[1..] {
            bias.push(Array2::<f64>::random((*shape, 1), Uniform::new(0., 3.)));
        }
        NeuralNet {
            weights,
            bias,
            activation: T::new(),
        }
    }
}

impl<T> NeuralNet<T>
where
    T: Activation,
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
                    unact.map(|x| self.activation.act(*x))
                }
            };
            all_activations.push(acts.clone());
        }
        all_activations.pop(); //Don't need the output twice!
        Ok(NetState {
            input,
            activations: all_activations,
            output: acts,
        })
    }
    /// Back-propagate the error from a training case and generate a new network with adjusted
    /// weights.
    pub fn back_prop(
        &self,
        mut state: NetState,
        actual: Array2<f64>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut activations = state.activations;
        //Calculate the first layer:
        let a_o = state.output;
        let de_dao = a_o - actual;
        let a_k: Array2<f64> = activations
            .pop()
            .ok_or_else(|| -> Box<dyn Error> { "Activation Vector empty!?".into() })?;
        let de_dwok = de_dao.dot(&a_k.t());
        println!("{:?}", de_dwok);
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
    pub input: Array2<f64>,
    pub activations: Vec<Array2<f64>>,
    pub output: Array2<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new() {
        let new: NeuralNet::<Relu> = NeuralNet::new_ones(vec![3, 4, 4, 1]);
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
            activation: crate::activations::Relu,
        };
        let inputs = array![[3.0], [4.0]];
        let soln = net.solve_fwd(inputs).unwrap();
        assert_eq!(soln.output, array![[21.0]])
    }
}

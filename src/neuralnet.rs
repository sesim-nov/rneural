use ndarray::{Array2,Array, array};

#[derive(Debug)]
pub struct NeuralNet<T>
where
    T: Fn(f64) -> f64,
{
    pub weights: Vec<Array2<f64>>,
    pub bias: Vec<Array2<f64>>,
    pub activation: T
}


impl NeuralNet<fn(f64) -> f64>
{
    // TODO: Maybe make all fields private and provided specified-weights constructor? 
    fn new_ones(nodes: Vec<usize>) -> Self{
        let mut weights = Vec::new();
        for shape in nodes.windows(2){
            weights.push(Array2::<f64>::ones((shape[0], shape[1])));
        }
        let mut bias = Vec::new();
        for shape in &nodes[1..]{
            bias.push(Array2::<f64>::ones((*shape, 1)));
        }
        NeuralNet{
            weights,
            bias, 
            activation: crate::activations::relu
        }
    }
}

impl<T> NeuralNet<T>
where
    T: Fn(f64) -> f64,
{
    // TODO: Appropriately error if a network is invalid (wrong weight matrix shapes, for example). 
    fn solve_fwd(&self, input: Array2<f64>) -> Result<Array2<f64>, &str>{
        let mut acts = input.clone();
        let num_weights = self.weights.len();
        for (i, weight) in self.weights.iter().enumerate() 
        {
            acts = 
            {
                let unact = weight.dot(&acts) + &self.bias[i];
                // Avoid applying activation to the output layer. 
                if num_weights - 1 == i 
                {
                    unact
                }
                else 
                {
                    unact.map(|x| {
                        (self.activation)(*x)
                    })
                }
            };
        }
        Ok(acts)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn test_new(){
        let new = NeuralNet::new_ones(vec![3,4,4,1]);
        assert_eq!(new.weights[0].shape(), &[3,4]);
        assert_eq!(new.weights[1].shape(), &[4,4]);
        assert_eq!(new.weights[2].shape(), &[4,1]);
        assert_eq!(new.bias[0].shape(), &[4,1]);
        assert_eq!(new.bias[1].shape(), &[4,1]);
        assert_eq!(new.bias[2].shape(), &[1,1]);

    }
    #[test]
    fn test_solve(){
        let net = NeuralNet{
            weights: vec![array![[1.0,1.0],[2.0,2.0]], array![[1.0,1.0]]],
            bias: vec![array![[0.0],[0.0]], array![[0.0]]],
            activation: crate::activations::relu
        };
        let inputs = array![[3.0],[4.0]];
        let soln = net.solve_fwd(inputs);
        assert_eq!(soln, Ok(array![[21.0]]))
    }
}

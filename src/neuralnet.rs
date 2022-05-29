use ndarray::{Array2,Array};

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
}

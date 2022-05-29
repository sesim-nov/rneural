use ndarray::{Array2};
pub struct NeuralNet<T>
where
    T: Fn(f64) -> f64,
{
    pub weights: Vec<Array2<f64>>,
    pub bias: Vec<Array2<f64>>,
    pub activation: T
}

package mlem

import (
	"math"
	"math/rand"
)

// Predictor is a generic prediction algorithm that is able to make
// predictions and adjust its parameters.
type Predictor interface {
	inputs() int
	parameters() int
	Predict([]float64) []float64
	change([]float64) []neuronGradient
	updateParameters([]neuronGradient)
}

// Neuron is the most basic set of calculations of a neural network.
type Neuron struct {
	Weights []float64
	Bias    float64
}

// When you see []neuronGradient, think "layer gradients."
type neuronGradient struct {
	Bias       float64
	Parameters []float64
	Inputs     []float64
}

// NewNeuron creates a neuron with random weights.
func NewNeuron(inputs int, src rand.Source) *Neuron {
	r := rand.New(src)
	n := &Neuron{Weights: make([]float64, inputs), Bias: r.Float64()}
	for i := range n.Weights {
		n.Weights[i] = (r.Float64() - 0.5) / 2
	}
	return n
}

func (n *Neuron) inputs() int { return len(n.Weights) }

func (n *Neuron) parameters() int { return 1 + len(n.Weights) }

// Predict goes through a set of calculations, also known as `feed forward`.
func (n *Neuron) Predict(inputs []float64) []float64 {
	v := 0.0
	for i, input := range inputs {
		v += input * n.Weights[i]
	}
	return []float64{v + n.Bias}
}

func (n *Neuron) change(inputs []float64) []neuronGradient {
	y := neuronGradient{
		Bias:       1,
		Parameters: inputs,
		Inputs:     make([]float64, len(n.Weights)),
	}
	for i, w := range n.Weights {
		y.Inputs[i] = w
	}
	return []neuronGradient{y}
}

func (n *Neuron) updateParameters(toAdd []neuronGradient) {
	neuronGradient := toAdd[0]
	n.Bias += neuronGradient.Bias
	for i, p := range neuronGradient.Parameters {
		n.Weights[i] += p
	}
}

// Loss calculates how wrong a set of predictions. It utilizes the Mean
// Square Error method.
func Loss(p Predictor, data [][]float64) float64 {
	mse := 0.0
	for _, row := range data {
		inputs, output := row[:p.inputs()], row[p.inputs()]
		err := output - p.Predict(inputs)[0]
		mse += err * err
	}
	return mse / float64(len(data))
}

func emptyNetworkGradient(p Predictor, data [][]float64) []neuronGradient {
	layerGradients := p.change(data[0][:p.inputs()])
	for j, neuronGradients := range layerGradients {
		layerGradients[j].Bias = 0
		for k := range neuronGradients.Parameters {
			layerGradients[j].Parameters[k] = 0
			layerGradients[j].Inputs[k] = 0
		}
	}
	return layerGradients
}

func lossChange(p Predictor, data [][]float64) []neuronGradient {
	finalGradients := p.change(data[0][:p.inputs()])
	// finalGradients := emptyNetworkGradient(p, data)
	for _, row := range data {
		inputs, output := row[:p.inputs()], row[p.inputs()]
		prediction := p.Predict(inputs)[0]
		layerGradients := p.change(inputs)
		for j, neuronGradients := range layerGradients {
			finalGradients[j].Bias = (prediction - output)
			for k, gradient := range neuronGradients.Parameters {
				finalGradients[j].Parameters[k] += (prediction - output) * gradient
			}
		}
	}
	for i, layerGradients := range finalGradients {
		finalGradients[i].Bias *= (2 / float64(len(data)))
		for k, gradient := range layerGradients.Parameters {
			finalGradients[i].Parameters[k] = gradient * (2 / float64(len(data)))
		}
	}
	return finalGradients
}

// Step receives the gradients to optimize loss, multiplies them by the
// learning rate, and updates the weights in the network.
func Step(p Predictor, data [][]float64, LearningRate float64) {
	layerGradients := lossChange(p, data)
	for j, neuronGradients := range layerGradients {
		layerGradients[j].Bias *= -LearningRate
		for k, gradient := range neuronGradients.Parameters {
			layerGradients[j].Parameters[k] = -LearningRate * gradient
		}
	}
	p.updateParameters(layerGradients)
}

// Layer is a collection of neurons that share the same input.
type Layer struct {
	Neurons    []*Neuron
	Activation *ActivationFunction
}

// NewLayer creates a collection of neurons with random weights.
func NewLayer(inputs, outputs int, act *ActivationFunction, src rand.Source) *Layer {
	neurons := make([]*Neuron, outputs)
	for i := range neurons {
		neurons[i] = NewNeuron(inputs, src)
	}
	return &Layer{Neurons: neurons, Activation: act}
}

func (l *Layer) inputs() int { return l.Neurons[0].inputs() }

func (l *Layer) outputs() int    { return len(l.Neurons) }
func (l *Layer) parameters() int { return len(l.Neurons) * l.Neurons[0].parameters() }

// Predict performs a feed forward for a collection of neurons.
func (l *Layer) Predict(inputs []float64) []float64 {
	var predictionSet []float64
	for _, neuron := range l.Neurons {
		basicPrediction := neuron.Predict(inputs)[0]
		actPrediction := l.Activation.Value(basicPrediction)
		predictionSet = append(predictionSet, actPrediction)
	}
	return predictionSet
}

func (l *Layer) change(inputs []float64) []neuronGradient {
	y := []neuronGradient{}
	for _, neuron := range l.Neurons {
		prediction := neuron.Predict(inputs)[0]
		actGradient := l.Activation.Gradient(prediction)
		neuronGradients := neuron.change(inputs)[0]
		for i, v := range neuronGradients.Parameters {
			neuronGradients.Parameters[i] = v * actGradient
		}
		y = append(y, neuronGradients)
	}
	return y
}

func (l *Layer) updateParameters(toAdd []neuronGradient) {
	layerGradients := toAdd // []NeuronGradient
	for i, neuron := range l.Neurons {
		singleNeuron := []neuronGradient{layerGradients[i]}
		neuron.updateParameters(singleNeuron)
	}
}

// ActivationFunction transforms the output of a basic neuron.
type ActivationFunction struct {
	Name     string
	Value    func(float64) float64
	Gradient func(float64) float64
}

var (
	// Identity is an activation function that does nothing.
	Identity = &ActivationFunction{
		"Identity",
		func(x float64) float64 { return x },
		func(x float64) float64 { return 1 },
	}
	sigmoid = func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	// Sigmoid is crazy math.
	Sigmoid = &ActivationFunction{
		"Sigmoid",
		sigmoid,
		func(x float64) float64 { return sigmoid(x) * (1 - sigmoid(x)) },
	}
	// RELU is a rectified linear unit activation function.
	RELU = &ActivationFunction{
		"RELU",
		func(x float64) float64 {
			if x > 0 {
				return x
			}
			return 0.0
		},
		func(x float64) float64 {
			if x > 0 {
				return 1.0
			}
			return 0.0
		},
	}
	activationsByName = map[string]*ActivationFunction{
		Identity.Name: Identity,
		Sigmoid.Name:  Sigmoid,
		RELU.Name:     RELU,
	}
)

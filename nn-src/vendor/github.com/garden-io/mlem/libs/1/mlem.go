package mlem

import (
	"math/rand"
)

// Predictor is a generic prediction algorithm that is able to make
// predictions and adjust its parameters.
type Predictor interface {
	inputs() int
	parameters() int
	Predict([]float64) float64
	change([]float64) neuronGradient
	updateParameters(neuronGradient)
}

// Neuron is the most basic set of calculations of a neural network.
type Neuron struct {
	Weights []float64
	Bias    float64
}

// When you see []neuronGradient, think "layer gradients."
// When you see [][]neuronGradient, think "network gradients."
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
func (n *Neuron) Predict(inputs []float64) float64 {
	v := 0.0
	for i, input := range inputs {
		v += input * n.Weights[i]
	}
	return v + n.Bias
}

func (n *Neuron) change(inputs []float64) neuronGradient {
	y := neuronGradient{
		Bias:       1,
		Parameters: inputs,
		Inputs:     make([]float64, len(n.Weights)),
	}
	for i, w := range n.Weights {
		y.Inputs[i] = w
	}
	return y
}

func (n *Neuron) updateParameters(toAdd neuronGradient) {
	neuronGradient := toAdd
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
		err := output - p.Predict(inputs)
		mse += err * err
	}
	return mse / float64(len(data))
}

func emptyNetworkGradient(p Predictor, data [][]float64) neuronGradient {
	neuronGradients := p.change(data[0][:p.inputs()])
	neuronGradients.Bias = 0
	for k := range neuronGradients.Parameters {
		neuronGradients.Parameters[k] = 0
		neuronGradients.Inputs[k] = 0
	}
	return neuronGradients
}

func lossChange(p Predictor, data [][]float64) neuronGradient {
	finalGradients := p.change(data[0][:p.inputs()])
	// finalGradients := emptyNetworkGradient(p, data)
	for _, row := range data {
		inputs, output := row[:p.inputs()], row[p.inputs()]
		prediction := p.Predict(inputs)
		neuronGradients := p.change(inputs)
		finalGradients.Bias = (prediction - output)
		for k, gradient := range neuronGradients.Parameters {
			finalGradients.Parameters[k] += (prediction - output) * gradient
		}
	}
	finalGradients.Bias *= (2 / float64(len(data)))
	for k, gradient := range finalGradients.Parameters {
		finalGradients.Parameters[k] = gradient * (2 / float64(len(data)))

	}
	return finalGradients
}

// Step receives the gradients to optimize loss, multiplies them by the
// learning rate, and updates the weights in the network.
func Step(p Predictor, data [][]float64, LearningRate float64) {
	neuronGradients := lossChange(p, data)
	neuronGradients.Bias *= -LearningRate
	for k, gradient := range neuronGradients.Parameters {
		neuronGradients.Parameters[k] = -LearningRate * gradient
	}
	p.updateParameters(neuronGradients)
}

package mlem

import (
	"fmt"
	"math"
	"math/rand"
)

// Predictor is a generic prediction algorithm that is able to make
// predictions and adjust its parameters.
type Predictor interface {
	inputs() int
	parameters() int
	Predict([]float64) []float64
	change([]float64) [][]neuronGradient
	updateParameters([][]neuronGradient)
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
func (n *Neuron) Predict(inputs []float64) []float64 {
	v := 0.0
	for i, input := range inputs {
		v += input * n.Weights[i]
	}
	return []float64{v + n.Bias}
}

func (n *Neuron) change(inputs []float64) [][]neuronGradient {
	y := neuronGradient{
		Bias:       1,
		Parameters: inputs,
		Inputs:     make([]float64, len(n.Weights)),
	}
	for i, w := range n.Weights {
		y.Inputs[i] = w
	}
	return [][]neuronGradient{[]neuronGradient{y}}
}

func (n *Neuron) updateParameters(toAdd [][]neuronGradient) {
	neuronGradient := toAdd[0][0]
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

func emptyNetworkGradient(p Predictor, data [][]float64) [][]neuronGradient {
	networkGradients := p.change(data[0][:p.inputs()])
	for i, layerGradients := range networkGradients {
		for j, neuronGradients := range layerGradients {
			networkGradients[i][j].Bias = 0
			for k := range neuronGradients.Parameters {
				networkGradients[i][j].Parameters[k] = 0
				networkGradients[i][j].Inputs[k] = 0
			}
		}
	}
	return networkGradients
}

func lossChange(p Predictor, data [][]float64) [][]neuronGradient {
	finalGradients := p.change(data[0][:p.inputs()])
	// finalGradients := emptyNetworkGradient(p, data)
	for _, row := range data {
		inputs, output := row[:p.inputs()], row[p.inputs()]
		prediction := p.Predict(inputs)[0]
		networkGradients := p.change(inputs)
		for i, layerGradients := range networkGradients {
			for j, neuronGradients := range layerGradients {
				finalGradients[i][j].Bias = (prediction - output)
				for k, gradient := range neuronGradients.Parameters {
					finalGradients[i][j].Parameters[k] += (prediction - output) * gradient
				}
			}
		}
	}
	for i, layerGradients := range finalGradients {
		for j, neuronGradients := range layerGradients {
			finalGradients[i][j].Bias *= (2 / float64(len(data)))
			for k, gradient := range neuronGradients.Parameters {
				finalGradients[i][j].Parameters[k] = gradient * (2 / float64(len(data)))
			}
		}
	}
	return finalGradients
}

// Step receives the gradients to optimize loss, multiplies them by the
// learning rate, and updates the weights in the network.
func Step(p Predictor, data [][]float64, LearningRate float64) {
	networkGradients := lossChange(p, data)
	for i, layerGradients := range networkGradients {
		for j, neuronGradients := range layerGradients {
			networkGradients[i][j].Bias *= -LearningRate
			for k, gradient := range neuronGradients.Parameters {
				networkGradients[i][j].Parameters[k] = -LearningRate * gradient
			}
		}
	}
	p.updateParameters(networkGradients)
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

func (l *Layer) change(inputs []float64) [][]neuronGradient {
	y := [][]neuronGradient{[]neuronGradient{}}
	for _, neuron := range l.Neurons {
		prediction := neuron.Predict(inputs)[0]
		actGradient := l.Activation.Gradient(prediction)
		neuronGradients := neuron.change(inputs)[0][0]
		for i, v := range neuronGradients.Parameters {
			neuronGradients.Parameters[i] = v * actGradient
		}
		y[0] = append(y[0], neuronGradients)
	}
	return y
}

func (l *Layer) updateParameters(toAdd [][]neuronGradient) {
	layerGradients := toAdd[0] // []NeuronGradient
	for i, neuron := range l.Neurons {
		singleNeuron := [][]neuronGradient{[]neuronGradient{layerGradients[i]}}
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

// Network is a collection of layers.
type Network struct {
	Layers []*Layer
}

// NewNetwork creates a network utilizing the layers specified as arguments.
// Example:
// 	network, err = NewNetwork(
// 		NewLayer(inputs, 4, mlem.RELU, src),
// 		NewLayer(4, 1, mlem.Identity, src),
// 	)
func NewNetwork(layers ...*Layer) (*Network, error) {
	for i := 1; i < len(layers); i++ {
		if layers[i-1].outputs() != layers[i].inputs() {
			return nil, fmt.Errorf("layer %d generates %d outputs, but next layer only accepts %d inputs",
				i, layers[i-1].outputs(), layers[i].inputs())
		}
	}
	return &Network{layers}, nil
}

func (nn *Network) inputs() int { return nn.Layers[0].inputs() }

func (nn *Network) outputs() int { return nn.Layers[len(nn.Layers)-1].outputs() }

// Predict performs a feed forward throughout every layer in the network.
func (nn *Network) Predict(inputs []float64) []float64 {
	values := inputs
	for _, layer := range nn.Layers {
		values = layer.Predict(values)
	}
	return values
}

func (nn *Network) parameters() int {
	count := 0
	for _, l := range nn.Layers {
		count += l.parameters()
	}
	return count
}

func (nn *Network) change(inputs []float64) [][]neuronGradient {
	// We start with a feed forward of the network that stores the input
	// values each layer will receive. We'll need them to calulate the
	// layer gradients below.
	predictions := make([][]float64, len(nn.Layers))
	values := inputs
	for i, l := range nn.Layers {
		predictions[i] = append(predictions[i], values...)
		values = l.Predict(values)
	}

	networkGradients := [][]neuronGradient{}
	for l := len(nn.Layers) - 1; l >= 0; l-- {
		layerGradients := nn.Layers[l].change(predictions[l])[0]
		// If it's the final layer, do nothing.
		if l == len(nn.Layers)-1 {
			networkGradients = [][]neuronGradient{layerGradients}
			continue
		}
		// As we count down the list of layers, we push the previous ones on
		// top of the later ones.
		networkGradients = append([][]neuronGradient{layerGradients}, networkGradients...)
		// For every neuron N, we sum all Nth input gradients of the next
		// layer, then multiply this sum by the local gradients of the
		// current neuron.
		for neuronNumber, currentNeuronGrads := range networkGradients[0] {
			var nextLayerGradSum float64
			for _, nextNeuronGrads := range networkGradients[1] {
				nextLayerGradSum += nextNeuronGrads.Inputs[neuronNumber]
			}
			networkGradients[0][neuronNumber].Bias *= nextLayerGradSum
			for param := range currentNeuronGrads.Parameters {
				networkGradients[0][neuronNumber].Parameters[param] *= nextLayerGradSum
			}
		}
	}
	return networkGradients
}

func (nn *Network) updateParameters(toAdd [][]neuronGradient) {
	for i, l := range nn.Layers {
		l.updateParameters([][]neuronGradient{toAdd[i]})
	}
}

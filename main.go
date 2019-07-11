package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	mlem "github.com/garden-io/mlem/libs/3"
)

func main() {

	dataset, err := ReadCSV("normalize/data.csv")
	if err != nil {
		log.Fatalf("could not read dataset: %v", err)
	}

	inputs := len(dataset[0]) - 1
	src := rand.NewSource(time.Now().Unix())
	learningRate := 0.25

	var network mlem.Predictor

	// Neuron!
	// network = mlem.NewNeuron(inputs, src)

	// Layer!
	// network = mlem.NewLayer(inputs, 1, mlem.Identity, src)

	// Network!
	network, err = mlem.NewNetwork(
		mlem.NewLayer(inputs, 4, mlem.RELU, src),
		mlem.NewLayer(4, 1, mlem.Identity, src),
	)
	if err != nil {
		fmt.Println(err.Error())
		return
	}

	cycles := 200
	for i := 0; i < cycles; i++ {
		mserr := mlem.Loss(network, dataset)
		mlem.Step(network, dataset, learningRate)
		if i%(cycles/20) == 0 {
			fmt.Printf("%d cost: %.6f, %.6f\n", i, mserr, learningRate)
		}
	}

	for i := 0; i < 5; i++ {
		out := network.Predict((dataset[i][:inputs]))
		fmt.Printf("car %d expected price %.4f; got %.4f\n", i, dataset[i][inputs], out)
	}
}

// ReadCSV reads the given file and returns a matrix of float64s.
func ReadCSV(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("could not open file: %v", err)
	}
	defer f.Close()
	var rows [][]float64
	r := csv.NewReader(f)
	for {
		cols, err := r.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("could not read line %d: %v", len(rows)+1, err)
		}
		var row []float64
		for i, col := range cols {
			v, err := strconv.ParseFloat(col, 64)
			if err != nil {
				return nil, fmt.Errorf("could not parse value %q on line %d column %d: %v", col, len(rows)+1, i+1, err)
			}
			row = append(row, v)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

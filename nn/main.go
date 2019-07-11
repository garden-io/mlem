package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"time"

	mlem "github.com/garden-io/mlem/libs/3"
	"github.com/gorilla/websocket"
)

// Values is the values, yo.
type Values struct {
	Progress int
	Loss     []float64
	Table    []string
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

func main() {
	port := "8080"
	if len(os.Args) > 2 {
		port = os.Args[2]
	}
	http.HandleFunc("/nn", serve)
	fmt.Println(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}

func nn(dataset [][]float64, w http.ResponseWriter, r *http.Request, conn *websocket.Conn) {
	inputs := len(dataset[0]) - 1
	src := rand.NewSource(time.Now().Unix())
	learningRate := 0.25
	toSend := Values{
		Progress: 100,
		Loss:     []float64{},
		Table:    []string{},
	}

	// Neuron!
	// var network mlem.Predictor
	// network = mlem.NewNeuron(inputs, src)

	// Layer!
	// var network mlem.Predictor
	// network = mlem.NewLayer(inputs, 1, mlem.Identity, src)

	// Network!
	network, err := mlem.NewNetwork(
		mlem.NewLayer(inputs, 5, mlem.RELU, src),
		mlem.NewLayer(5, 1, mlem.Identity, src),
	)
	if err != nil {
		fmt.Println(err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	cycles := 500
	for i := 0; i < cycles; i++ {
		mserr := mlem.Loss(network, dataset)
		mlem.Step(network, dataset, learningRate)
		if i%(cycles/64) == 0 {
			toSend.Loss = append(toSend.Loss, mserr)
		}
		if i%(cycles/20) == 0 {
			toSend.Progress = ((i / (cycles / 20)) + 1) * 5
			toSend.Table = []string{
				// fmt.Sprintf("%f", network.Predict(dataset[0][:inputs])),
				// fmt.Sprintf("%f", network.Predict(dataset[1][:inputs])),
				// fmt.Sprintf("%f", network.Predict(dataset[2][:inputs])),
				// fmt.Sprintf("%f", network.Predict(dataset[3][:inputs])),
				// fmt.Sprintf("%f", network.Predict(dataset[4][:inputs])),
				fmt.Sprintf("%f", network.Predict(dataset[0][:inputs])[0]),
				fmt.Sprintf("%f", network.Predict(dataset[1][:inputs])[0]),
				fmt.Sprintf("%f", network.Predict(dataset[2][:inputs])[0]),
				fmt.Sprintf("%f", network.Predict(dataset[3][:inputs])[0]),
				fmt.Sprintf("%f", network.Predict(dataset[4][:inputs])[0]),
			}
			send(toSend, w, r, conn)
			fmt.Printf("%d cost: %.6f, %.6f\n", i, mserr, learningRate)

		}
		// learningRate *= 0.99
	}
	samples := [][]float64{
		{-1.6476, 1, -1.6319, 0.62007},
		{-1.2473, 1, -1.3261, 0.25555},
		{-1.1132, 1, -0.7147, 0.25794},
		{-0.0716, 1, 1.73111, 0.07407},
		{1.26784, 1, 1.42538, 0.05615},
	}
	for i, sample := range samples {
		out := network.Predict((sample[:inputs]))
		fmt.Printf("car %d expected price %.4f; got %.4f\n", i, sample[inputs], out)
	}
}

func send(toSend Values, w http.ResponseWriter, r *http.Request, conn *websocket.Conn) {
	js, err := json.Marshal(toSend)
	if err != nil {
		fmt.Println(err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err = conn.WriteMessage(1, js); err != nil {
		fmt.Println(err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func serve(w http.ResponseWriter, r *http.Request) {
	upgrader.CheckOrigin = func(r *http.Request) bool { return true }
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Println(err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	fmt.Println("Connected!")
	for {
		msgType, msg, err := conn.ReadMessage()
		if err != nil {
			fmt.Println(err.Error())
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if string(msg) == "ping" {
			fmt.Printf("%s nn sent: %s\n", conn.RemoteAddr(), string(msg))
			if err = conn.WriteMessage(msgType, []byte("pong")); err != nil {
				fmt.Println(err.Error())
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		} else {
			var dataset struct {
				Dataset [][]float64
			}
			if err := json.Unmarshal(msg, &dataset); err != nil {
				fmt.Println(err.Error())
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			nn(dataset.Dataset, w, r, conn)
		}
	}
}

package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

type stuff struct {
	cars    [][]float64
	top5    [][]string
	payload []byte
	nope    error
}

var theUglyGlobalVariable stuff

func main() {
	var err error
	theUglyGlobalVariable.cars = data("data.csv")
	bla := struct {
		Top5    [][]string
		Dataset [][]float64
	}{theUglyGlobalVariable.top5, theUglyGlobalVariable.cars}
	theUglyGlobalVariable.payload, err = json.Marshal(bla)
	if err != nil {
		theUglyGlobalVariable.nope = err
	}
	_ = theUglyGlobalVariable.payload

	port := "8080"
	if len(os.Args) > 1 {
		port = os.Args[1]
	}
	http.HandleFunc("/data", serve)
	fmt.Println(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}

func data(file string) [][]float64 {
	var output [][]float64
	var total int
	fileContent, err := os.Open(file)
	if err != nil {
		theUglyGlobalVariable.nope = err
	}
	reader := csv.NewReader(bufio.NewReader(fileContent))
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		kmsInt, _ := strconv.ParseFloat(record[0], 64)
		ageInt, _ := strconv.ParseFloat(record[2], 64)
		priceInt, _ := strconv.ParseFloat(record[3], 64)
		output = append(output, []float64{kmsInt, 1.0, ageInt, priceInt})
		if total < 5 {
			theUglyGlobalVariable.top5 = append(theUglyGlobalVariable.top5, []string{record[0], record[2], record[3]})
		}
		total++
	}
	fmt.Println("Loaded", total, "entries.")
	return output
}

func serve(w http.ResponseWriter, r *http.Request) {

	if theUglyGlobalVariable.nope != nil {
		fmt.Println(theUglyGlobalVariable.nope.Error())
		http.Error(w, theUglyGlobalVariable.nope.Error(), http.StatusInternalServerError)
		return
	}

	upgrader.CheckOrigin = func(r *http.Request) bool { return true }

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Println(err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err = conn.WriteMessage(1, theUglyGlobalVariable.payload); err != nil {
		fmt.Println(err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	fmt.Printf("Content sent to: %s\n", conn.RemoteAddr())

	for {
		msgType, msg, err := conn.ReadMessage()
		if err != nil {
			fmt.Println(err.Error())
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		fmt.Printf("%s normal sent: %s\n", conn.RemoteAddr(), string(msg))
		if string(msg) == "ping" {
			if err = conn.WriteMessage(msgType, []byte("pong")); err != nil {
				fmt.Println(err.Error())
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
	}
}

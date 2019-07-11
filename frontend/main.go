package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
)

func main() {
	port := "8080"
	if len(os.Args) > 1 {
		port = os.Args[1]
	}
	http.HandleFunc("/", serve)
	fmt.Println(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}

func serve(w http.ResponseWriter, r *http.Request) {
	absPath, err := filepath.Abs("./index.html")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	file, err := ioutil.ReadFile(absPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write(file)
}

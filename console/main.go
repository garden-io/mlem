package main

import (
	"log"
	"math"
	"time"
	"fmt"
	"net/http"
	"encoding/json"
	"os"
	"github.com/gorilla/websocket"

	ui "github.com/gizak/termui/v3"
	"github.com/gizak/termui/v3/widgets"
)

type values struct {
	MainText string
	ConnNN string
	ConnNormalize string
	ConnFrontend string
	ConnConsole string
	Progress int
	Loss []float64
	Table [][]string
}

var myValues = values{
	MainText: "Learn Neural Networks With Go - Not Math!\ngithub.com/garden-io/mlem\n@ellenkorbes",
	ConnNN: "[• neural network](fg:red)", 
	ConnNormalize: "[• data](fg:red)",
	ConnFrontend: "[• frontend](fg:red)",
	ConnConsole: "[• console](fg:green)",
	Progress: 0,
	Loss: []float64{},
	Table: [][]string{
		[]string{"input: km", "input: years", "price", "predicted"},
		[]string{"0", "0", "0", "0"},
		[]string{"0", "0", "0", "0"},
		[]string{"0", "0", "0", "0"},
		[]string{"0", "0", "0", "0"},
		[]string{"0", "0", "0", "0"},
	},
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

func doTheThing() {
	port := "8090"
	if len(os.Args) > 1 {
		port = os.Args[1]
	}
	http.HandleFunc("/console", serve)
	http.ListenAndServe(fmt.Sprintf(":%s", port), nil)
}

func serve(w http.ResponseWriter, r *http.Request) {
	upgrader.CheckOrigin = func(r *http.Request) bool { return true }
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		myValues.MainText = err.Error()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for {
		msgType, msg, err := conn.ReadMessage()
		if err != nil {
			myValues.MainText = err.Error()
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if string(msg) == "ping" {
			if err = conn.WriteMessage(msgType, []byte("pong")); err != nil {
				myValues.MainText = err.Error()
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		} else {
			myValues.ConnNN = string(string(msg))
			if err := json.Unmarshal(msg, &myValues); err != nil {
				myValues.MainText = err.Error()
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
	}
}

func main() {
	
	go doTheThing()
	if err := ui.Init(); err != nil {
		log.Fatalf("failed to initialize termui: %v", err)
	}
	defer ui.Close()

	p := widgets.NewParagraph()
	p.Title = "hi!"
	p.Text = "Learn Neural Networks With Go - Not Math!\ngithub.com/garden-io/mlem\n@ellenkorbes"
	p.SetRect(0, 0, 50, 5)
	p.TextStyle.Fg = ui.ColorCyan
	p.TitleStyle.Fg = ui.ColorCyan
	p.BorderStyle.Fg = ui.ColorWhite

	listData := []string{
		"",
		myValues.ConnNN,
		myValues.ConnNormalize,
		myValues.ConnFrontend,
		myValues.ConnConsole,
	}

	l := widgets.NewList()
	l.Title = "services"
	l.Rows = listData
	l.SetRect(51, 0, 75, 8)
	l.TextStyle.Fg = ui.ColorWhite
	l.TitleStyle.Fg = ui.ColorCyan

	g := widgets.NewGauge()
	g.Title = "progress"
	g.Percent = 50
	g.SetRect(0, 5, 50, 8)
	g.BarColor = ui.ColorMagenta
	g.BorderStyle.Fg = ui.ColorWhite
	g.TitleStyle.Fg = ui.ColorCyan

	sinData := (func() []float64 {
		n := 220
		ps := make([]float64, n)
		for i := range ps {
			ps[i] = 1 + math.Sin(float64(i)/5)
		}
		return ps
	})()

	lc := widgets.NewPlot()
	lc.Title = " loss: "
	lc.Data = make([][]float64, 1)
	lc.Data[0] = sinData
	lc.SetRect(0, 9, 75, 24)
	lc.AxesColor = ui.ColorWhite
	lc.LineColors[0] = ui.ColorCyan
	lc.Marker = widgets.MarkerDot
	lc.TitleStyle.Fg = ui.ColorCyan
	lc.DataLabels = []string{"epochs", "loss"}

	p2 := widgets.NewParagraph()
	p2.Text = "garden.io"
	p2.Border = false
	p2.SetRect(64, 38, 75, 39)
	p2.TextStyle.Fg = ui.ColorMagenta

	table1 := widgets.NewTable()
	table1.Rows = myValues.Table
	table1.TextStyle = ui.NewStyle(ui.ColorWhite)
	table1.TextAlignment = ui.AlignCenter
	table1.RowStyles[0] = ui.NewStyle(ui.ColorWhite, ui.ColorBlack, ui.ModifierBold)
	table1.SetRect(0, 25, 75, 38)

	draw := func(count int) {
		g.Percent = myValues.Progress
		table1.Rows = myValues.Table
		p.Text = myValues.MainText
		l.Rows = []string{
			"",
			myValues.ConnNN,
			myValues.ConnNormalize,
			myValues.ConnFrontend,
			myValues.ConnConsole,
		}
		lc.Data[0] = myValues.Loss
		if len(myValues.Loss) > 0 { lc.Title = fmt.Sprint(" loss: ", myValues.Loss[len(myValues.Loss)-1], " ")}
		ui.Render(p, l, g, lc, p2, table1)
	}

	tickerCount := 1
	draw(tickerCount)
	tickerCount++
	uiEvents := ui.PollEvents()
	ticker := time.NewTicker(time.Second).C
	for {
		select {
		case e := <-uiEvents:
			switch e.ID {
			case "q", "<C-c>":
				return
			}
		case <-ticker:
			// updateParagraph(tickerCount)
			draw(tickerCount)
			tickerCount++
		}
	}
}

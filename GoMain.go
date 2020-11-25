package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	_ "net/http/pprof"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"

	"time"

	"gopkg.in/cheggaaa/pb.v1"
)

var (
	epochs     = flag.Int("epochs", 5, "Number of epochs")
	dataset    = flag.String("dataset", "train", " \"train\" or \"valid\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 32, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "./mnist/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type GoNet struct {
	g          *gorgonia.ExprGraph
	w0, w1, w2 *gorgonia.Node

	out     *gorgonia.Node
	predVal gorgonia.Value
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func newGoNet(g *gorgonia.ExprGraph) *GoNet {
	
	w0 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(784, 300), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(300, 100), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(100, 10), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	return &GoNet{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
	}
}

func (m *GoNet) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2}
}

func (m *GoNet) fwd(x *gorgonia.Node) (err error) {
	var l0, l1, l2 *gorgonia.Node
	var l0dot, l1dot *gorgonia.Node
	l0 = x

	
	if l0dot, err = gorgonia.Mul(l0, m.w0); err != nil {
		return errors.Wrap(err, "Unable to multiply l0 and w0")
	}

	l0dot := gorgonia.Must(gorgonia.Mul(l0, m.w0))	
	l1 = gorgonia.Must(gorgonia.Rectify(l0dot))

	if l1dot, err = gorgonia.Mul(l1, m.w1); err != nil {
		return errors.Wrap(err, "Unable to multiply l1 and w1")
	}
	l2 = gorgonia.Must(gorgonia.Rectify(l1dot))

	var out *gorgonia.Node
	if out, err = gorgonia.Mul(l2, m.w2); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w2")
	}
	m.out, err = gorgonia.SoftMax(out)
	gorgonia.Read(m.out, &m.predVal)
	return

}

const pixelRange = 255

func reversePixelWeight(px float64) byte {
	// return byte((pixelRange*px - pixelRange) / 0.9)
	return byte(pixelRange*math.Min(0.99, math.Max(0.01, px)) - pixelRange)
}

func visualizeRow(x []float64) *image.Gray {
	// since this is a square, we can take advantage of that
	l := len(x)
	side := int(math.Sqrt(float64(l)))
	r := image.Rect(0, 0, side, side)
	img := image.NewGray(r)

	pix := make([]byte, l)
	for i, px := range x {
		pix[i] = reversePixelWeight(px)
	}
	img.Pix = pix

	return img
}

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(7945)

	// // intercept Ctrl+C
	// sigChan := make(chan os.Signal, 1)
	// signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// doneChan := make(chan bool, 1)

	var inputs, targets tensor.Tensor
	var err error

	// load our data set
	trainOn := *dataset
	if inputs, targets, err = mnist.Load(trainOn, loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples := inputs.Shape()[0]
	bs := *batchsize

	g := gorgonia.NewGraph()
	x := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"))

	m := newGoNet(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	losses, err := gorgonia.HadamardProd(m.out, y)
	if err != nil {
		log.Fatal(err)
	}
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	prog, locmap, err := gorgonia.Compile(g)
	fmt.Println(prog)
	fmt.Println(locmap)

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))

	log.Printf("Starting Training...")

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ { 
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}

			if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
				log.Fatal("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		bar.Update()
		log.Printf("Epoch %d | cost %v", i, costVal)
	}
	bar.Finish()

	log.Printf("Run Tests")

	// load our test set
	if inputs, targets, err = mnist.Load("test", loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples = inputs.Shape()[0]
	bs = *batchsize
	batches = numExamples / bs

	bar = pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)
	bar.Prefix(fmt.Sprintf("Epoch Test"))
	bar.Set(0)
	bar.Start()
	for b := 0; b < batches; b++ {
		start := b * bs
		end := start + bs
		if start >= numExamples {
			break
		}
		if end > numExamples {
			end = numExamples
		}

		var xVal, yVal tensor.Tensor
		if xVal, err = inputs.Slice(sli{start, end}); err != nil {
			log.Fatal("Unable to slice x")
		}

		if yVal, err = targets.Slice(sli{start, end}); err != nil {
			log.Fatal("Unable to slice y")
		}
		if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
			log.Fatal("Unable to reshape %v", err)
		}

		gorgonia.Let(x, xVal)
		gorgonia.Let(y, yVal)
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at epoch test: %v", err)
		}

		arrayOutput := m.predVal.Data().([]float64)
		yOutput := tensor.New(tensor.WithShape(bs, 10), tensor.WithBacking(arrayOutput))

		for j := 0; j < xVal.Shape()[0]; j++ {
			rowT, _ := xVal.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			yRowT, _ := yVal.Slice(sli{j, j + 1})
			yRow := yRowT.Data().([]float64)
			var rowLabel int
			var yRowHigh float64

			for k := 0; k < 10; k++ {
				if k == 0 {
					rowLabel = 0
					yRowHigh = yRow[k]
				} else if yRow[k] > yRowHigh {
					rowLabel = k
					yRowHigh = yRow[k]
				}
			}

			predRowT, _ := yOutput.Slice(sli{j, j + 1})
			predRow := predRowT.Data().([]float64)
			var rowGuess int
			var predRowHigh float64

			for k := 0; k < 10; k++ {
				if k == 0 {
					rowGuess = 0
					predRowHigh = predRow[k]
				} else if predRow[k] > predRowHigh {
					rowGuess = k
					predRowHigh = predRow[k]
				}
			}

			f, _ := os.OpenFile(fmt.Sprintf("images/%d - %d - %d - %d.jpg", b, j, rowLabel, rowGuess), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
			f.Close()
		}

		arrayOutput = m.predVal.Data().([]float64)
		yOutput = tensor.New(tensor.WithShape(bs, 10), tensor.WithBacking(arrayOutput))

		file, err := os.OpenFile(fmt.Sprintf("%d.csv", b), os.O_CREATE|os.O_WRONLY, 0777)
		if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
			log.Fatal("Unable to create csv", err)
		}
		defer file.Close()
		var matrixToWrite [][]string

		for j := 0; j < yOutput.Shape()[0]; j++ {
			rowT, _ := yOutput.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)
			var rowToWrite []string

			for k := 0; k < 10; k++ {
				rowToWrite = append(rowToWrite, strconv.FormatFloat(row[k], 'f', 6, 64))
			}
			matrixToWrite = append(matrixToWrite, rowToWrite)
		}

		csvWriter := csv.NewWriter(file)
		csvWriter.WriteAll(matrixToWrite)
		csvWriter.Flush()

		vm.Reset()
		bar.Increment()
	}
	log.Printf("Epoch Test | cost %v", costVal)

}
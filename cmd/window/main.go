package main

import (
	"image/color"
	"fmt"
	"log"
	"math"
	"errors"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/examples/resources/fonts"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/neural"
	"github.com/zachbeta/go_inverted_pendulum/pkg/reward"
)

const (
	screenWidth  = 800
	screenHeight = 600
	scale       = 100.0 // pixels per meter
	
	// Network visualization constants
	networkPanelX = 10
	networkPanelY = 10
	networkPanelWidth = 250
	networkPanelHeight = 200
	nodeRadius = 15
	
	// Network layer positions
	inputLayerX = networkPanelX + 40
	hiddenLayerX = networkPanelX + networkPanelWidth/2
	outputLayerX = networkPanelX + networkPanelWidth - 40
)

var (
	mplusNormalFont font.Face
)

func init() {
	tt, err := opentype.Parse(fonts.MPlus1pRegular_ttf)
	if err != nil {
		log.Fatal(err)
	}
	const dpi = 72
	mplusNormalFont, err = opentype.NewFace(tt, &opentype.FaceOptions{
		Size:    12,
		DPI:     dpi,
		Hinting: font.HintingFull,
	})
	if err != nil {
		log.Fatal(err)
	}
}

type Game struct {
	pendulum *env.Pendulum
	network  *neural.Network
	logger   *log.Logger
	cartImg  *ebiten.Image
	bobImg   *ebiten.Image
	prevState env.State // Store previous state for reward calculation
	episodes  int       // Track number of training episodes
	ticks     int       // Track ticks within current episode
	maxTicks  int       // Track best episode length
	lastHiddenActivation float64 // Store last hidden layer activation
}

func NewGame(pendulum *env.Pendulum, logger *log.Logger) *Game {
	// Create cart image
	cartImg := ebiten.NewImage(50, 30)
	cartImg.Fill(color.RGBA{100, 100, 255, 255})

	// Create pendulum bob image
	bobImg := ebiten.NewImage(20, 20)
	bobImg.Fill(color.RGBA{255, 100, 100, 255})

	return &Game{
		pendulum: pendulum,
		network:  neural.NewNetwork(),
		logger:   logger,
		cartImg:  cartImg,
		bobImg:   bobImg,
		prevState: pendulum.GetState(),
		episodes: 0,
		ticks: 0,
		maxTicks: 0,
		lastHiddenActivation: 0,
	}
}

func (g *Game) Update() error {
	// Check for window close
	if ebiten.IsWindowBeingClosed() {
		return errors.New("window closed")
	}

	// Get current state
	state := g.pendulum.GetState()
	
	// Get force from network and store hidden activation
	force, hiddenActivation := g.network.ForwardWithActivation(state)
	g.lastHiddenActivation = hiddenActivation
	g.logger.Printf("Episode %d Tick %d: Network force: %.2f N for angle: %.2f rad\n", 
		g.episodes, g.ticks, force, state.AngleRadians)
	
	// Apply force and get new state
	newState, err := g.pendulum.Step(force)
	if err != nil {
		g.logger.Printf("Episode %d ended after %d ticks with error: %v\n", 
			g.episodes, g.ticks, err)
		
		// Update max ticks if this was the best episode
		if g.ticks > g.maxTicks {
			g.maxTicks = g.ticks
			g.logger.Printf("New best episode! Max ticks: %d\n", g.maxTicks)
		}
		
		// Calculate final reward for this episode
		finalReward := reward.Calculate(g.prevState, state)
		g.network.Update(finalReward)
		g.logger.Printf("Final reward for episode %d: %.4f\n", g.episodes, finalReward)
		
		// Reset pendulum for next episode
		config := g.pendulum.GetConfig()
		g.pendulum = env.NewPendulum(config, g.logger)
		g.prevState = g.pendulum.GetState()
		g.episodes++
		g.ticks = 0
		g.logger.Printf("Starting episode %d\n", g.episodes)
		return nil
	}

	// Calculate and apply reward for this step
	stepReward := reward.Calculate(g.prevState, newState)
	g.network.Update(stepReward)
	
	// Update previous state and increment ticks
	g.prevState = newState
	g.ticks++
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Get current state
	state := g.pendulum.GetState()
	
	// Draw track
	trackY := float64(screenHeight) * 0.7
	ebitenutil.DrawLine(screen, 0, trackY, float64(screenWidth), trackY, color.White)
	
	// Calculate cart position in screen coordinates
	cartWidth := float64(g.cartImg.Bounds().Dx())
	cartHeight := float64(g.cartImg.Bounds().Dy())
	cartX := float64(screenWidth)/2 + state.CartPosition*scale
	
	// Draw cart
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(cartX-cartWidth/2, trackY-cartHeight)
	screen.DrawImage(g.cartImg, op)
	
	// Calculate pendulum end point
	pendulumLength := g.pendulum.GetConfig().Length * scale
	endX := cartX + pendulumLength*math.Sin(state.AngleRadians)
	endY := trackY - cartHeight/2 + pendulumLength*math.Cos(state.AngleRadians)
	
	// Draw pendulum
	ebitenutil.DrawLine(screen,
		cartX,
		trackY-cartHeight/2,
		endX,
		endY,
		color.RGBA{255, 100, 100, 255})
	
	// Draw pendulum bob
	bobWidth := float64(g.bobImg.Bounds().Dx())
	bobHeight := float64(g.bobImg.Bounds().Dy())
	op = &ebiten.DrawImageOptions{}
	op.GeoM.Translate(endX-bobWidth/2, endY-bobHeight/2)
	screen.DrawImage(g.bobImg, op)
	
	// Draw debug info
	weights := g.network.GetWeights()
	debugText := fmt.Sprintf(
		"Training Progress\n"+
		"Episode: %d\n"+
		"Current Ticks: %d\n"+
		"Best Episode: %d ticks\n\n"+
		"State:\n"+
		"  Cart Position: %.2f m\n"+
		"  Cart Velocity: %.2f m/s\n"+
		"  Angle: %.2f rad (%.1f°)\n"+
		"  Angular Vel: %.2f rad/s\n\n"+
		"Network Weights:\n"+
		"  Angle: %.4f\n"+
		"  Angular Vel: %.4f\n"+
		"  Bias: %.4f",
		g.episodes,
		g.ticks,
		g.maxTicks,
		state.CartPosition,
		state.CartVelocity,
		state.AngleRadians,
		state.AngleRadians * 180 / math.Pi,
		state.AngularVel,
		weights[0], weights[1], weights[2])
	ebitenutil.DebugPrint(screen, debugText)

	// Draw neural network visualization
	g.drawNetworkVisualization(screen, state)
}

func (g *Game) drawNetworkVisualization(screen *ebiten.Image, state env.State) {
	// Draw panel background
	ebitenutil.DrawRect(screen, 
		float64(networkPanelX), 
		float64(networkPanelY), 
		float64(networkPanelWidth), 
		float64(networkPanelHeight), 
		color.RGBA{40, 40, 40, 230})

	// Calculate node positions
	centerY := float64(networkPanelY + networkPanelHeight/2)
	
	// Input layer positions (spread vertically)
	angleY := centerY - 25
	velocityY := centerY + 25
	
	// Hidden layer position (center)
	hiddenY := centerY
	
	// Output layer positions
	outputY := centerY

	// Draw layer labels
	text.Draw(screen, "Inputs", mplusNormalFont, inputLayerX-25, networkPanelY+20, color.White)
	text.Draw(screen, "Hidden", mplusNormalFont, hiddenLayerX-25, networkPanelY+20, color.White)
	text.Draw(screen, "Output", mplusNormalFont, outputLayerX-25, networkPanelY+20, color.White)

	// Get network weights and current force
	weights := g.network.GetWeights()
	angleWeight := weights[0]
	angularVelWeight := weights[1]
	currentForce := g.pendulum.GetLastForce()

	// Draw connections with thickness based on weight magnitude
	drawWeightedConnection(screen, 
		float64(inputLayerX), angleY,
		float64(hiddenLayerX), hiddenY,
		angleWeight)
	drawWeightedConnection(screen, 
		float64(inputLayerX), velocityY,
		float64(hiddenLayerX), hiddenY,
		angularVelWeight)
	drawWeightedConnection(screen, 
		float64(hiddenLayerX), hiddenY,
		float64(outputLayerX), outputY,
		1.0)

	// Calculate node colors based on activations
	angleColor := getActivationColor(state.AngleRadians / math.Pi)
	velocityColor := getActivationColor(state.AngularVel / 10.0)
	hiddenColor := getActivationColor(g.lastHiddenActivation)
	outputColor := getActivationColor(currentForce / 5.0)

	// Draw input nodes with labels
	drawNode(screen, float64(inputLayerX), angleY, "θ", angleColor)
	drawNode(screen, float64(inputLayerX), velocityY, "ω", velocityColor)
	text.Draw(screen, fmt.Sprintf("%.2f°", state.AngleRadians*180/math.Pi), 
		mplusNormalFont, inputLayerX+20, int(angleY)+5, color.White)
	text.Draw(screen, fmt.Sprintf("%.2f rad/s", state.AngularVel), 
		mplusNormalFont, inputLayerX+20, int(velocityY)+5, color.White)

	// Draw hidden node with activation
	drawNode(screen, float64(hiddenLayerX), hiddenY, "H", hiddenColor)
	text.Draw(screen, fmt.Sprintf("%.2f", g.lastHiddenActivation), 
		mplusNormalFont, hiddenLayerX-20, int(hiddenY)-20, color.White)

	// Draw output node with force
	drawNode(screen, float64(outputLayerX), outputY, "F", outputColor)
	text.Draw(screen, fmt.Sprintf("%.1f N", currentForce), 
		mplusNormalFont, outputLayerX-20, int(outputY)-20, color.White)

	// Draw network title and description
	text.Draw(screen, "Neural Network State", mplusNormalFont, 
		networkPanelX+5, networkPanelY+15, color.White)
}

func drawWeightedConnection(screen *ebiten.Image, x1, y1, x2, y2, weight float64) {
	// Normalize weight to determine line color
	normalizedWeight := math.Abs(weight) / 3.0 // Max weight is 3.0
	if normalizedWeight > 1.0 {
		normalizedWeight = 1.0
	}

	// Create color based on weight sign
	var lineColor color.Color
	if weight >= 0 {
		lineColor = color.RGBA{
			uint8(100 + 155*normalizedWeight),
			uint8(100 + 155*normalizedWeight),
			255,
			255,
		}
	} else {
		lineColor = color.RGBA{
			255,
			uint8(100 + 155*normalizedWeight),
			uint8(100 + 155*normalizedWeight),
			255,
		}
	}

	ebitenutil.DrawLine(screen, x1, y1, x2, y2, lineColor)
}

func drawNode(screen *ebiten.Image, x, y float64, label string, nodeColor color.Color) {
	ebitenutil.DrawCircle(screen, x, y, float64(nodeRadius), nodeColor)
	ebitenutil.DrawCircle(screen, x, y, float64(nodeRadius)-1, color.Black)
	
	// Draw label
	bound, _ := font.BoundString(mplusNormalFont, label)
	w := (bound.Max.X - bound.Min.X).Ceil()
	h := (bound.Max.Y - bound.Min.Y).Ceil()
	text.Draw(screen, label, mplusNormalFont, 
		int(x)-w/2, 
		int(y)+h/2, 
		color.White)
}

func getActivationColor(activation float64) color.Color {
	// Clamp activation to [-1, 1]
	if activation > 1.0 {
		activation = 1.0
	} else if activation < -1.0 {
		activation = -1.0
	}

	// Convert to [0, 1] range
	normalized := (activation + 1.0) / 2.0

	// Create color gradient from blue (negative) to white (zero) to red (positive)
	if activation >= 0 {
		return color.RGBA{
			uint8(255 * normalized),
			uint8(100 + 155 * (1-normalized)),
			uint8(100 + 155 * (1-normalized)),
			255,
		}
	} else {
		return color.RGBA{
			uint8(100 + 155 * (1+normalized)),
			uint8(100 + 155 * (1+normalized)),
			255 * uint8(1-normalized),
			255,
		}
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 800, 600
}

func main() {
	// Create logger
	logger := log.Default()
	
	// Create pendulum with default config
	config := env.NewDefaultConfig()
	pendulum := env.NewPendulum(config, logger)
	
	// Create game
	game := NewGame(pendulum, logger)
	
	// Set up window
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Inverted Pendulum - Network Control")
	
	// Run game
	if err := ebiten.RunGame(game); err != nil {
		if err.Error() == "window closed" {
			logger.Println("Window closed normally")
		} else {
			logger.Printf("Game error: %v\n", err)
		}
	}
}

package main

import (
	"image/color"
	"fmt"
	"log"
	"math"
	"errors"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/neural"
)

const (
	screenWidth  = 800
	screenHeight = 600
	scale       = 100.0 // pixels per meter
)

type Game struct {
	pendulum *env.Pendulum
	network  *neural.Network
	logger   *log.Logger
	cartImg  *ebiten.Image
	bobImg   *ebiten.Image
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
	}
}

func (g *Game) Update() error {
	// Check for window close
	if ebiten.IsWindowBeingClosed() {
		return errors.New("window closed")
	}

	// Get current state
	state := g.pendulum.GetState()
	
	// Get force from network
	force := g.network.Forward(state)
	g.logger.Printf("Network force: %.2f N for angle: %.2f rad\n", force, state.AngleRadians)
	
	// Apply force
	_, err := g.pendulum.Step(force)
	if err != nil {
		g.logger.Printf("Error stepping simulation: %v\n", err)
		// Reset pendulum near center with current angle
		config := g.pendulum.GetConfig()
		g.pendulum = env.NewPendulum(config, g.logger)
		// The pendulum's NewPendulum will initialize with default state
	}

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
		"Network Control\n\n"+
		"State:\n"+
		"  Cart Position: %.2f m\n"+
		"  Cart Velocity: %.2f m/s\n"+
		"  Angle: %.2f rad (%.1f°)\n"+
		"  Angular Vel: %.2f rad/s\n\n"+
		"Network Weights:\n"+
		"  Angle: %.4f\n"+
		"  Angular Vel: %.4f\n"+
		"  Bias: %.4f",
		state.CartPosition,
		state.CartVelocity,
		state.AngleRadians,
		state.AngleRadians * 180 / math.Pi,
		state.AngularVel,
		weights[0], weights[1], weights[2])
	ebitenutil.DebugPrint(screen, debugText)
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

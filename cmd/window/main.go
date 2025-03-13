package main

import (
	"image/color"
	"fmt"
	"log"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
)

const (
	screenWidth  = 800
	screenHeight = 600
	scale       = 100.0 // pixels per meter
)

type Game struct {
	pendulum *env.Pendulum
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
		logger:   logger,
		cartImg:  cartImg,
		bobImg:   bobImg,
	}
}

func (g *Game) Update() error {
	// For now, just apply a constant force
	if ebiten.IsKeyPressed(ebiten.KeyRight) {
		_, err := g.pendulum.Step(5.0)
		if err != nil {
			g.logger.Printf("Error stepping simulation: %v\n", err)
		}
	} else if ebiten.IsKeyPressed(ebiten.KeyLeft) {
		_, err := g.pendulum.Step(-5.0)
		if err != nil {
			g.logger.Printf("Error stepping simulation: %v\n", err)
		}
	} else {
		_, err := g.pendulum.Step(0)
		if err != nil {
			g.logger.Printf("Error stepping simulation: %v\n", err)
		}
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
	debugText := fmt.Sprintf(
		"Cart Position: %.2f m\n"+
		"Cart Velocity: %.2f m/s\n"+
		"Pendulum Angle: %.2f rad\n"+
		"Angular Velocity: %.2f rad/s",
		state.CartPosition,
		state.CartVelocity,
		state.AngleRadians,
		state.AngularVel)
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
	ebiten.SetWindowTitle("Inverted Pendulum Simulation")
	
	// Run game
	if err := ebiten.RunGame(game); err != nil {
		logger.Fatal(err)
	}
}

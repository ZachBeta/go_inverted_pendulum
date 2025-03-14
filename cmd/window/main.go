package main

import (
	"errors"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/examples/resources/fonts"
	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/neural"
	"github.com/zachbeta/go_inverted_pendulum/pkg/render"
	"github.com/zachbeta/go_inverted_pendulum/pkg/reward"
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
	drawer   *render.Drawer
	prevState env.State // Store previous state for reward calculation
	episodes  int       // Track number of training episodes
	ticks     int       // Track ticks within current episode
	maxTicks  int       // Track best episode length
	lastHiddenActivation float64 // Store last hidden layer activation
}

func NewGame(pendulum *env.Pendulum, logger *log.Logger) *Game {
	return &Game{
		pendulum: pendulum,
		network:  neural.NewNetwork(),
		logger:   logger,
		drawer:   render.NewDrawer(mplusNormalFont),
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
	g.drawer.Draw(screen, g.pendulum, g.network, g.episodes, g.ticks, g.maxTicks, g.lastHiddenActivation)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return render.ScreenWidth, render.ScreenHeight
}

func main() {
	// Set up logger
	logger := log.Default()

	// Create pendulum with initial configuration
	config := env.Config{
		CartMass:     5.0,   // kg
		PendulumMass: 1.0,   // kg
		Length:       1.0,    // m
		Gravity:      9.81,   // m/s²
		MaxForce:     10.0,   // N
		DeltaTime:    0.016,  // s (60 fps)
		TrackLength:  4.0,    // m
	}
	pendulum := env.NewPendulum(config, logger)

	// Create and run game
	game := NewGame(pendulum, logger)
	ebiten.SetWindowSize(render.ScreenWidth, render.ScreenHeight)
	ebiten.SetWindowTitle("Inverted Pendulum Neural Network")
	if err := ebiten.RunGame(game); err != nil {
		logger.Fatal(err)
	}
}

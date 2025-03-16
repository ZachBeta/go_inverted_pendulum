package main

import (
	"errors"
	"os"
	"path/filepath"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/examples/resources/fonts"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"github.com/zachbeta/go_inverted_pendulum/pkg/ensemble"
	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/logger"
	"github.com/zachbeta/go_inverted_pendulum/pkg/render"
)

var (
	mplusNormalFont font.Face
)

func init() {
	tt, err := opentype.Parse(fonts.MPlus1pRegular_ttf)
	if err != nil {
		panic(err)
	}
	const dpi = 72
	mplusNormalFont, err = opentype.NewFace(tt, &opentype.FaceOptions{
		Size:    12,
		DPI:     dpi,
		Hinting: font.HintingFull,
	})
	if err != nil {
		panic(err)
	}
}

type Game struct {
	ensemble     *ensemble.Ensemble
	drawer       *render.Drawer
	logger       *logger.Logger
	networkPath  string   // Path to save/load network state
}

func NewGame(gameLogger *logger.Logger) *Game {
	// Create pendulum configuration
	pendulumConfig := env.Config{
		CartMass:     5.0,   // kg
		PendulumMass: 1.0,   // kg
		Length:       1.0,    // m
		Gravity:      9.81,   // m/s²
		MaxForce:     10.0,   // N
		DeltaTime:    0.016,  // s (60 fps)
		TrackLength:  4.0,    // m
	}
	
	// Create ensemble configuration
	ensembleConfig := ensemble.NewDefaultConfig()
	ensembleConfig.NetworkCount = 10 // Train 10 networks simultaneously
	
	// Create ensemble
	ensemble := ensemble.NewEnsemble(ensembleConfig, pendulumConfig, gameLogger.GetStandardLogger())
	
	// Set up network save path in user's home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		gameLogger.Error("Failed to get home directory, using current directory: %v", err)
		homeDir = "."
	}
	networkPath := filepath.Join(homeDir, ".inverted_pendulum", "network.json")

	return &Game{
		ensemble:     ensemble,
		drawer:       render.NewDrawer(mplusNormalFont),
		logger:       gameLogger,
		networkPath:  networkPath,
	}
}

func (g *Game) Update() error {
	// Check for window close
	if ebiten.IsWindowBeingClosed() {
		return errors.New("window closed")
	}

	// Handle network save/load
	if inpututil.IsKeyJustPressed(ebiten.KeyS) {
		bestNetwork := g.ensemble.GetBestNetwork()
		if err := bestNetwork.Network.SaveToFile(g.networkPath); err != nil {
			g.logger.Error("Failed to save network: %v", err)
		} else {
			g.logger.Info("Best network saved to %s", g.networkPath)
		}
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyL) {
		// Load network into the best network instance
		bestNetwork := g.ensemble.GetBestNetwork()
		if err := bestNetwork.Network.LoadFromFile(g.networkPath); err != nil {
			g.logger.Error("Failed to load network: %v", err)
		} else {
			g.logger.Info("Network loaded from %s", g.networkPath)
		}
	}

	// Update all networks in the ensemble
	if err := g.ensemble.Step(); err != nil {
		g.logger.Error("Ensemble step error: %v", err)
	}
	
	// Get best network for visualization
	bestNetwork := g.ensemble.GetBestNetwork()
	
	// Update drawer with latest training stats
	g.drawer.UpdateTrainingStats(bestNetwork.Trainer)
	
	// Update ensemble statistics
	g.drawer.UpdateEnsembleStats(g.ensemble.GetAllNetworkStats())

	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Get the best network for visualization
	bestNetwork := g.ensemble.GetBestNetwork()
	
	// Draw the pendulum and network visualization
	g.drawer.Draw(
		screen, 
		bestNetwork.Pendulum, 
		bestNetwork.Network, 
		bestNetwork.Trainer, 
		bestNetwork.Episodes, 
		bestNetwork.CurrentTicks, 
		bestNetwork.MaxTicks, 
		bestNetwork.LastHiddenActivation,
	)
	
	// Draw ensemble statistics
	g.drawer.DrawEnsembleStats(screen)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return render.ScreenWidth, render.ScreenHeight
}

func main() {
	// Set up custom logger
	// Show INFO and ERROR on console, but log everything to file
	gameLogger, err := logger.NewLogger(logger.INFO, logger.DEBUG)
	if err != nil {
		panic(err)
	}
	defer gameLogger.Close()
	
	gameLogger.Info("Starting Inverted Pendulum Neural Network Ensemble")

	// Create and run game
	game := NewGame(gameLogger)
	ebiten.SetWindowSize(render.ScreenWidth, render.ScreenHeight)
	ebiten.SetWindowTitle("Inverted Pendulum Neural Network Ensemble")
	
	if err := ebiten.RunGame(game); err != nil {
		gameLogger.Fatal("Game error: %v", err)
	}
}

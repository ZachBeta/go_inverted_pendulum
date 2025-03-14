package render

import (
	"fmt"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font"
	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/neural"
)

const (
	ScreenWidth  = 800
	ScreenHeight = 600
	Scale       = 100.0 // pixels per meter
	
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

type Drawer struct {
	cartImg  *ebiten.Image
	bobImg   *ebiten.Image
	font     font.Face
}

func NewDrawer(font font.Face) *Drawer {
	// Create cart image
	cartImg := ebiten.NewImage(50, 30)
	cartImg.Fill(color.RGBA{100, 100, 255, 255})

	// Create pendulum bob image
	bobImg := ebiten.NewImage(20, 20)
	bobImg.Fill(color.RGBA{255, 100, 100, 255})

	return &Drawer{
		cartImg:  cartImg,
		bobImg:   bobImg,
		font:     font,
	}
}

func (d *Drawer) Draw(screen *ebiten.Image, pendulum *env.Pendulum, network *neural.Network, episodes, ticks, maxTicks int, lastHiddenActivation float64) {
	state := pendulum.GetState()
	
	// Draw track
	trackY := float64(ScreenHeight) * 0.7
	ebitenutil.DrawLine(screen, 0, trackY, float64(ScreenWidth), trackY, color.White)
	
	// Calculate cart position in screen coordinates
	cartWidth := float64(d.cartImg.Bounds().Dx())
	cartHeight := float64(d.cartImg.Bounds().Dy())
	cartX := float64(ScreenWidth)/2 + state.CartPosition*Scale
	
	// Draw cart
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(cartX-cartWidth/2, trackY-cartHeight)
	screen.DrawImage(d.cartImg, op)
	
	// Calculate pendulum end point
	pendulumLength := pendulum.GetConfig().Length * Scale
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
	bobWidth := float64(d.bobImg.Bounds().Dx())
	bobHeight := float64(d.bobImg.Bounds().Dy())
	op = &ebiten.DrawImageOptions{}
	op.GeoM.Translate(endX-bobWidth/2, endY-bobHeight/2)
	screen.DrawImage(d.bobImg, op)
	
	// Draw debug info
	weights := network.GetWeights()
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
		episodes,
		ticks,
		maxTicks,
		state.CartPosition,
		state.CartVelocity,
		state.AngleRadians,
		state.AngleRadians*180/math.Pi,
		state.AngularVel,
		weights[0],
		weights[1],
		weights[2])
	
	text.Draw(screen, debugText, d.font, 550, 30, color.White)
	
	d.drawNetworkVisualization(screen, state, network, lastHiddenActivation)
}

func (d *Drawer) drawNetworkVisualization(screen *ebiten.Image, state env.State, network *neural.Network, lastHiddenActivation float64) {
	// Draw panel background
	ebitenutil.DrawRect(screen, float64(networkPanelX), float64(networkPanelY), 
		float64(networkPanelWidth), float64(networkPanelHeight), color.RGBA{40, 40, 40, 255})

	// Calculate node positions
	inputY1 := float64(networkPanelY) + float64(networkPanelHeight)/3
	inputY2 := float64(networkPanelY) + 2*float64(networkPanelHeight)/3
	hiddenY := float64(networkPanelY) + float64(networkPanelHeight)/2
	outputY := float64(networkPanelY) + float64(networkPanelHeight)/2

	// Draw connections with weights
	weights := network.GetWeights()
	d.drawWeightedConnection(screen, inputLayerX, inputY1, hiddenLayerX, hiddenY, weights[0])
	d.drawWeightedConnection(screen, inputLayerX, inputY2, hiddenLayerX, hiddenY, weights[1])
	d.drawWeightedConnection(screen, hiddenLayerX, hiddenY, outputLayerX, outputY, 1.0)

	// Draw nodes
	d.drawNode(screen, inputLayerX, inputY1, "θ", color.White)
	d.drawNode(screen, inputLayerX, inputY2, "ω", color.White)
	d.drawNode(screen, hiddenLayerX, hiddenY, "H", d.getActivationColor(lastHiddenActivation))
	d.drawNode(screen, outputLayerX, outputY, "F", color.White)

	// Draw node values
	text.Draw(screen, fmt.Sprintf("%.2f", state.AngleRadians), 
		d.font, int(inputLayerX)+25, int(inputY1), color.White)
	text.Draw(screen, fmt.Sprintf("%.2f", state.AngularVel), 
		d.font, int(inputLayerX)+25, int(inputY2), color.White)
	text.Draw(screen, fmt.Sprintf("%.2f", lastHiddenActivation), 
		d.font, int(hiddenLayerX)+25, int(hiddenY), color.White)
}

func (d *Drawer) drawWeightedConnection(screen *ebiten.Image, x1, y1, x2, y2, weight float64) {
	// Calculate color based on weight
	var lineColor color.Color
	if weight > 0 {
		intensity := uint8(math.Min(255, weight*128))
		lineColor = color.RGBA{intensity, 255, intensity, 255}
	} else {
		intensity := uint8(math.Min(255, -weight*128))
		lineColor = color.RGBA{255, intensity, intensity, 255}
	}

	// Draw line
	ebitenutil.DrawLine(screen, x1, y1, x2, y2, lineColor)

	// Draw weight value
	midX := (x1 + x2) / 2
	midY := (y1 + y2) / 2
	weightText := fmt.Sprintf("%.2f", weight)
	bounds := text.BoundString(d.font, weightText)
	text.Draw(screen, weightText, d.font, 
		int(midX)-bounds.Dx()/2, int(midY), color.White)
}

func (d *Drawer) drawNode(screen *ebiten.Image, x, y float64, label string, nodeColor color.Color) {
	ebitenutil.DrawCircle(screen, x, y, nodeRadius, nodeColor)
	bounds := text.BoundString(d.font, label)
	text.Draw(screen, label, d.font, 
		int(x)-bounds.Dx()/2, int(y)+bounds.Dy()/2, color.Black)
}

func (d *Drawer) getActivationColor(activation float64) color.Color {
	if activation <= 0 {
		// Blue to white gradient for negative values
		intensity := uint8(math.Min(255, -activation*128))
		return color.RGBA{intensity, intensity, 255, 255}
	} else {
		// White to red gradient for positive values
		intensity := uint8(math.Min(255, activation*128))
		return color.RGBA{255, intensity, intensity, 255}
	}
}

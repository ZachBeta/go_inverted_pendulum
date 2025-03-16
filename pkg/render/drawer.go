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
	"github.com/zachbeta/go_inverted_pendulum/pkg/training"
)

const (
	ScreenWidth  = 1024
	ScreenHeight = 768
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
	
	// Top info panel
	topPanelHeight = 60
	
	// Bottom info panel
	bottomPanelHeight = 80
	
	// Weight history graph
	weightHistoryX = 10
	weightHistoryY = ScreenHeight - bottomPanelHeight - 180
	weightHistoryWidth = 250
	weightHistoryHeight = 180
	
	// Maximum number of weight history points to display
	maxHistoryPoints = 50
)

type Drawer struct {
	cartImg  *ebiten.Image
	bobImg   *ebiten.Image
	font     font.Face
	
	// Weight history tracking
	angleWeightHistory     []float64
	angularVelWeightHistory []float64
	biasWeightHistory      []float64
	
	// Performance metrics
	successRateHistory     []float64
	rewardHistory          []float64
	
	// Training stats
	episodeDurations       []int
	maxEpisodeDuration     int
	avgReward              float64
	successRate            float64
	learningRate           float64
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
		angleWeightHistory: make([]float64, 0, maxHistoryPoints),
		angularVelWeightHistory: make([]float64, 0, maxHistoryPoints),
		biasWeightHistory: make([]float64, 0, maxHistoryPoints),
		successRateHistory: make([]float64, 0, maxHistoryPoints),
		rewardHistory: make([]float64, 0, maxHistoryPoints),
		episodeDurations: make([]int, 0, maxHistoryPoints),
	}
}

func (d *Drawer) UpdateTrainingStats(trainer *training.Trainer) {
	if trainer == nil {
		return
	}
	
	// Get current training stats
	stats := trainer.GetTrainingStats()
	
	// Update learning rate
	if lr, ok := stats["learning_rate"].(float64); ok {
		d.learningRate = lr
	}
	
	// Update success rate
	if sr, ok := stats["success_rate"].(float64); ok {
		d.successRate = sr
		
		// Add to history, keeping only the most recent points
		d.successRateHistory = append(d.successRateHistory, sr)
		if len(d.successRateHistory) > maxHistoryPoints {
			d.successRateHistory = d.successRateHistory[1:]
		}
	}
	
	// Update average reward
	if ar, ok := stats["avg_reward"].(float64); ok {
		d.avgReward = ar
		
		// Add to history, keeping only the most recent points
		d.rewardHistory = append(d.rewardHistory, ar)
		if len(d.rewardHistory) > maxHistoryPoints {
			d.rewardHistory = d.rewardHistory[1:]
		}
	}
}

func (d *Drawer) Draw(screen *ebiten.Image, pendulum *env.Pendulum, network *neural.Network, trainer *training.Trainer, episodes, ticks, maxTicks int, lastHiddenActivation float64) {
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
	
	// Update weight history
	d.angleWeightHistory = append(d.angleWeightHistory, weights[0])
	d.angularVelWeightHistory = append(d.angularVelWeightHistory, weights[1])
	d.biasWeightHistory = append(d.biasWeightHistory, weights[2])
	
	// Keep history within max size
	if len(d.angleWeightHistory) > maxHistoryPoints {
		d.angleWeightHistory = d.angleWeightHistory[1:]
		d.angularVelWeightHistory = d.angularVelWeightHistory[1:]
		d.biasWeightHistory = d.biasWeightHistory[1:]
	}
	
	// Update episode durations if this is a new max
	if ticks > d.maxEpisodeDuration {
		d.maxEpisodeDuration = ticks
	}
	
	// Add current episode duration if episode ended
	if ticks == 0 && episodes > 0 && len(d.episodeDurations) < episodes {
		d.episodeDurations = append(d.episodeDurations, maxTicks)
		if len(d.episodeDurations) > maxHistoryPoints {
			d.episodeDurations = d.episodeDurations[1:]
		}
	}
	
	// Update training stats if trainer is provided
	if trainer != nil {
		d.UpdateTrainingStats(trainer)
	}
	
	// Draw top info panel
	d.drawTopInfoPanel(screen, episodes, ticks, maxTicks, state)
	
	// Draw bottom info panel
	d.drawBottomInfoPanel(screen, weights)
	
	// Draw network visualization
	d.drawNetworkVisualization(screen, state, network, lastHiddenActivation)
	
	// Draw weight history graph
	d.drawWeightHistoryGraph(screen)
}

func (d *Drawer) drawTopInfoPanel(screen *ebiten.Image, episodes, ticks, maxTicks int, state env.State) {
	// Draw panel background
	ebitenutil.DrawRect(screen, 0, 0, float64(ScreenWidth), float64(topPanelHeight), color.RGBA{40, 40, 40, 200})
	
	// Draw training progress
	trainingText := fmt.Sprintf(
		"Episode: %d | Current Ticks: %d | Best Episode: %d ticks | Success Rate: %.1f%% | Learning Rate: %.4f",
		episodes,
		ticks,
		maxTicks,
		d.successRate*100,
		d.learningRate)
	
	text.Draw(screen, trainingText, d.font, 10, 25, color.White)
	
	// Draw state info
	stateText := fmt.Sprintf(
		"Cart Position: %.2f m | Cart Velocity: %.2f m/s | Angle: %.2f rad (%.1f°) | Angular Velocity: %.2f rad/s",
		state.CartPosition,
		state.CartVelocity,
		state.AngleRadians,
		state.AngleRadians*180/math.Pi,
		state.AngularVel)
	
	text.Draw(screen, stateText, d.font, 10, 45, color.White)
}

func (d *Drawer) drawBottomInfoPanel(screen *ebiten.Image, weights []float64) {
	// Draw panel background
	ebitenutil.DrawRect(screen, 0, float64(ScreenHeight-bottomPanelHeight), float64(ScreenWidth), float64(bottomPanelHeight), color.RGBA{40, 40, 40, 200})
	
	// Draw network weights
	weightsText := fmt.Sprintf(
		"Network Weights: Angle: %.4f | Angular Velocity: %.4f | Bias: %.4f | Avg Reward: %.4f",
		weights[0],
		weights[1],
		weights[2],
		d.avgReward)
	
	text.Draw(screen, weightsText, d.font, 10, ScreenHeight-bottomPanelHeight+25, color.White)
	
	// Draw controls info
	controlsText := "Controls: Left/Right Arrow = Manual Force | S = Save Network | L = Load Network"
	text.Draw(screen, controlsText, d.font, 10, ScreenHeight-bottomPanelHeight+45, color.White)
	
	// Draw performance info
	performanceText := fmt.Sprintf("Episode Duration Trend: %d episodes tracked", len(d.episodeDurations))
	text.Draw(screen, performanceText, d.font, 10, ScreenHeight-bottomPanelHeight+65, color.White)
}

func (d *Drawer) drawNetworkVisualization(screen *ebiten.Image, state env.State, network *neural.Network, lastHiddenActivation float64) {
	// Draw panel background with title
	ebitenutil.DrawRect(screen, float64(networkPanelX), float64(networkPanelY + topPanelHeight + 10), 
		float64(networkPanelWidth), float64(networkPanelHeight), color.RGBA{40, 40, 40, 255})
	text.Draw(screen, "Network Architecture", d.font, 
		networkPanelX+5, networkPanelY+topPanelHeight+25, color.White)

	// Calculate node positions
	inputY1 := float64(networkPanelY + topPanelHeight + 10) + float64(networkPanelHeight)/3
	inputY2 := float64(networkPanelY + topPanelHeight + 10) + 2*float64(networkPanelHeight)/3
	hiddenY := float64(networkPanelY + topPanelHeight + 10) + float64(networkPanelHeight)/2
	outputY := float64(networkPanelY + topPanelHeight + 10) + float64(networkPanelHeight)/2

	// Draw connections with weights
	weights := network.GetWeights()
	d.drawWeightedConnection(screen, inputLayerX, inputY1, hiddenLayerX, hiddenY, weights[0])
	d.drawWeightedConnection(screen, inputLayerX, inputY2, hiddenLayerX, hiddenY, weights[1])
	d.drawWeightedConnection(screen, hiddenLayerX, hiddenY, outputLayerX, outputY, 1.0)

	// Draw bias connection
	d.drawBiasConnection(screen, hiddenLayerX, hiddenY, weights[2])

	// Draw nodes with activation colors
	d.drawNode(screen, inputLayerX, inputY1, "θ", d.getActivationColor(state.AngleRadians))
	d.drawNode(screen, inputLayerX, inputY2, "ω", d.getActivationColor(state.AngularVel))
	d.drawNode(screen, hiddenLayerX, hiddenY, "H", d.getActivationColor(lastHiddenActivation))
	
	// Calculate network output (force)
	force, _ := network.ForwardWithActivation(state)
	d.drawNode(screen, outputLayerX, outputY, "F", d.getActivationColor(force/5.0)) // Normalize force to [-1,1]

	// Draw node values
	text.Draw(screen, fmt.Sprintf("%.2f", state.AngleRadians), 
		d.font, int(inputLayerX)+25, int(inputY1), color.White)
	text.Draw(screen, fmt.Sprintf("%.2f", state.AngularVel), 
		d.font, int(inputLayerX)+25, int(inputY2), color.White)
	text.Draw(screen, fmt.Sprintf("%.2f", lastHiddenActivation), 
		d.font, int(hiddenLayerX)+25, int(hiddenY), color.White)
	text.Draw(screen, fmt.Sprintf("%.2f", force), 
		d.font, int(outputLayerX)+25, int(outputY), color.White)
}

func (d *Drawer) drawBiasConnection(screen *ebiten.Image, x, y, weight float64) {
	// Draw bias node
	biasX := x - 30
	biasY := y - 30
	d.drawNode(screen, biasX, biasY, "B", color.RGBA{200, 200, 200, 255})
	
	// Draw connection with weight
	d.drawWeightedConnection(screen, biasX, biasY, x, y, weight)
}

func (d *Drawer) drawWeightHistoryGraph(screen *ebiten.Image) {
	// Draw panel background with title
	ebitenutil.DrawRect(screen, float64(weightHistoryX), float64(weightHistoryY), 
		float64(weightHistoryWidth), float64(weightHistoryHeight), color.RGBA{40, 40, 40, 255})
	text.Draw(screen, "Weight History", d.font, 
		weightHistoryX+5, weightHistoryY+15, color.White)
	
	// Draw legend
	legendY := weightHistoryY + 30
	
	// Angle weight
	ebitenutil.DrawRect(screen, float64(weightHistoryX+10), float64(legendY-5), 10, 10, 
		color.RGBA{255, 100, 100, 255})
	text.Draw(screen, "Angle", d.font, weightHistoryX+25, legendY+5, color.White)
	
	// Angular velocity weight
	ebitenutil.DrawRect(screen, float64(weightHistoryX+70), float64(legendY-5), 10, 10, 
		color.RGBA{100, 255, 100, 255})
	text.Draw(screen, "Angular Vel", d.font, weightHistoryX+85, legendY+5, color.White)
	
	// Bias weight
	ebitenutil.DrawRect(screen, float64(weightHistoryX+170), float64(legendY-5), 10, 10, 
		color.RGBA{100, 100, 255, 255})
	text.Draw(screen, "Bias", d.font, weightHistoryX+185, legendY+5, color.White)
	
	// Draw weight history graphs
	graphWidth := weightHistoryWidth - 20
	graphHeight := 100
	graphX := weightHistoryX + 10
	graphY := weightHistoryY + 50
	
	// Draw zero line
	ebitenutil.DrawLine(screen, float64(graphX), float64(graphY+graphHeight/2), 
		float64(graphX+graphWidth), float64(graphY+graphHeight/2), 
		color.RGBA{150, 150, 150, 255})
	
	// Draw weight history lines if we have data
	if len(d.angleWeightHistory) > 1 {
		d.drawWeightGraph(screen, d.angleWeightHistory, graphX, graphY, graphWidth, graphHeight, 
			color.RGBA{255, 100, 100, 255})
		d.drawWeightGraph(screen, d.angularVelWeightHistory, graphX, graphY, graphWidth, graphHeight, 
			color.RGBA{100, 255, 100, 255})
		d.drawWeightGraph(screen, d.biasWeightHistory, graphX, graphY, graphWidth, graphHeight, 
			color.RGBA{100, 100, 255, 255})
	}
}

func (d *Drawer) drawProgressBar(screen *ebiten.Image, x, y, width, height int, value float64, barColor color.Color) {
	// Draw background
	ebitenutil.DrawRect(screen, float64(x), float64(y), float64(width), float64(height), 
		color.RGBA{80, 80, 80, 255})
	
	// Draw filled portion
	fillWidth := int(float64(width) * math.Max(0, math.Min(1, value)))
	if fillWidth > 0 {
		ebitenutil.DrawRect(screen, float64(x), float64(y), float64(fillWidth), float64(height), 
			barColor)
	}
	
	// Draw value text
	valueText := fmt.Sprintf("%.1f%%", value*100)
	bounds := text.BoundString(d.font, valueText)
	text.Draw(screen, valueText, d.font, 
		x+width/2-bounds.Dx()/2, y+height/2+bounds.Dy()/2, color.White)
}

func (d *Drawer) drawMiniGraph(screen *ebiten.Image, values []int, x, y, width, height int, lineColor color.Color) {
	if len(values) < 2 {
		return
	}
	
	// Find max value for scaling
	maxVal := 0
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}
	
	if maxVal == 0 {
		maxVal = 1 // Avoid division by zero
	}
	
	// Draw axes
	ebitenutil.DrawLine(screen, float64(x), float64(y+height), 
		float64(x+width), float64(y+height), color.RGBA{150, 150, 150, 255})
	ebitenutil.DrawLine(screen, float64(x), float64(y), 
		float64(x), float64(y+height), color.RGBA{150, 150, 150, 255})
	
	// Draw graph line
	pointSpacing := float64(width) / float64(len(values)-1)
	
	for i := 0; i < len(values)-1; i++ {
		x1 := float64(x) + float64(i)*pointSpacing
		y1 := float64(y+height) - float64(values[i])*float64(height)/float64(maxVal)
		x2 := float64(x) + float64(i+1)*pointSpacing
		y2 := float64(y+height) - float64(values[i+1])*float64(height)/float64(maxVal)
		
		ebitenutil.DrawLine(screen, x1, y1, x2, y2, lineColor)
	}
}

func (d *Drawer) drawWeightGraph(screen *ebiten.Image, values []float64, x, y, width, height int, lineColor color.Color) {
	if len(values) < 2 {
		return
	}
	
	// Find max absolute value for scaling
	maxVal := 0.0
	for _, v := range values {
		absV := math.Abs(v)
		if absV > maxVal {
			maxVal = absV
		}
	}
	
	if maxVal < 0.1 {
		maxVal = 0.1 // Minimum scale
	}
	
	// Draw graph line
	pointSpacing := float64(width) / float64(len(values)-1)
	
	for i := 0; i < len(values)-1; i++ {
		x1 := float64(x) + float64(i)*pointSpacing
		y1 := float64(y+height/2) - values[i]*float64(height/2)/maxVal
		x2 := float64(x) + float64(i+1)*pointSpacing
		y2 := float64(y+height/2) - values[i+1]*float64(height/2)/maxVal
		
		ebitenutil.DrawLine(screen, x1, y1, x2, y2, lineColor)
	}
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

	// Draw line with thickness based on absolute weight
	thickness := math.Max(1, math.Min(5, math.Abs(weight)*2))
	d.drawThickLine(screen, x1, y1, x2, y2, thickness, lineColor)

	// Draw weight value
	midX := (x1 + x2) / 2
	midY := (y1 + y2) / 2
	weightText := fmt.Sprintf("%.2f", weight)
	bounds := text.BoundString(d.font, weightText)
	text.Draw(screen, weightText, d.font, 
		int(midX)-bounds.Dx()/2, int(midY), color.White)
}

func (d *Drawer) drawThickLine(screen *ebiten.Image, x1, y1, x2, y2, thickness float64, lineColor color.Color) {
	// Draw main line
	ebitenutil.DrawLine(screen, x1, y1, x2, y2, lineColor)
	
	// Draw additional lines for thickness if needed
	if thickness > 1 {
		// Calculate perpendicular vector
		dx, dy := x2-x1, y2-y1
		length := math.Sqrt(dx*dx + dy*dy)
		if length > 0 {
			dx, dy = dx/length, dy/length
			perpX, perpY := -dy, dx
			
			// Draw additional lines
			for i := 1; i <= int(thickness); i++ {
				offset := float64(i) * 0.5
				ebitenutil.DrawLine(screen, 
					x1+perpX*offset, y1+perpY*offset, 
					x2+perpX*offset, y2+perpY*offset, 
					lineColor)
				ebitenutil.DrawLine(screen, 
					x1-perpX*offset, y1-perpY*offset, 
					x2-perpX*offset, y2-perpY*offset, 
					lineColor)
			}
		}
	}
}

func (d *Drawer) drawNode(screen *ebiten.Image, x, y float64, label string, nodeColor color.Color) {
	ebitenutil.DrawCircle(screen, x, y, nodeRadius, nodeColor)
	bounds := text.BoundString(d.font, label)
	text.Draw(screen, label, d.font, 
		int(x)-bounds.Dx()/2, int(y)+bounds.Dy()/2, color.Black)
}

func (d *Drawer) getActivationColor(activation float64) color.Color {
	// Normalize activation to [-1, 1] range if needed
	normalizedActivation := math.Max(-1, math.Min(1, activation))
	
	if normalizedActivation <= 0 {
		// Blue to white gradient for negative values
		intensity := uint8(math.Min(255, 255*(1+normalizedActivation)))
		return color.RGBA{intensity, intensity, 255, 255}
	} else {
		// White to red gradient for positive values
		intensity := uint8(math.Min(255, 255*(1-normalizedActivation)))
		return color.RGBA{255, intensity, intensity, 255}
	}
}

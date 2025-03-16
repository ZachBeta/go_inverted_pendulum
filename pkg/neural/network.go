// Package neural implements a simple three-node neural network
// for controlling the inverted pendulum, following Pezzza's
// original implementation approach with minimal parameters
package neural

import (
	"fmt"
	"log"
	"math"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/metrics"
)

// Network implements a three-node architecture:
// - Input: pendulum state (angle, angular velocity)
// - Hidden: single node for processing
// - Output: force decision (-5N to +5N)
type Network struct {
	// Weights for angle and angular velocity inputs
	angleWeight     float64
	angularVelWeight float64
	
	// Bias for hidden node
	bias float64
	
	// Learning parameters
	learningRate float64
	lastForce    float64 // Store last output for weight updates
	lastInputs   []float64 // Store last inputs for weight updates
	lastValue    float64 // Store last state value for TD learning

	// Progressive training parameters
	difficulty     float64  // Current difficulty level [0.0, 1.0]
	successRate    float64  // Recent success rate
	successWindow  []bool   // Window of recent success/failure
	windowSize     int      // Size of success tracking window
	progressThresh float64  // Success rate threshold for progression
	regressThresh  float64  // Success rate threshold for regression

	// Debug flag
	debug bool
	
	// Logger
	logger *log.Logger
	
	// Metrics logger for performance tracking
	metrics *metrics.Logger
	
	// Current episode and step tracking
	currentEpisode int
	currentStep    int
}

// NewNetwork creates a network with initialized weights
func NewNetwork() *Network {
	net := &Network{
		angleWeight:      6.0,  // Further increased for stronger response to angle
		angularVelWeight: 3.0,  // Further increased for stronger response to velocity
		bias:            0.0,  // Start with no bias
		learningRate:    0.05, // Learning rate for quick adaptation
		lastInputs:      make([]float64, 2),
		difficulty:      0.1,  // Start with low difficulty
		successRate:     0.0,  // Initial success rate
		windowSize:      100,  // Track last 100 attempts
		successWindow:   make([]bool, 0, 100),
		progressThresh:  0.8,  // Progress when 80% success rate
		regressThresh:  0.2,   // Regress when 20% success rate
		debug:           false, // Disable debug by default
		logger:          log.Default(),
		currentEpisode:  0,
		currentStep:     0,
	}
	
	// This console log remains as it's for initial creation and metrics logger isn't set yet
	net.logger.Printf("[Network] Created new network with initial weights: angle=%.4f, angularVelWeight=%.4f, bias=%.4f\n",
		net.angleWeight, net.angularVelWeight, net.bias)
	return net
}

// SetDebug enables or disables debug printing
func (n *Network) SetDebug(enabled bool) {
	n.debug = enabled
}

// SetLogger sets the logger for this network
func (n *Network) SetLogger(logger *log.Logger) {
	n.logger = logger
}

// SetMetricsLogger sets the metrics logger for performance tracking
func (n *Network) SetMetricsLogger(metricsLogger *metrics.Logger) {
	n.metrics = metricsLogger
	
	// Configure metrics logger to use sparse console output
	if metricsLogger != nil {
		// Log to console only every 5 episodes and every 500 steps
		metricsLogger.SetLogFrequency(5, 500)
		
		// Log initial weights to metrics database
		metricsLogger.LogWeights(n.angleWeight, n.angularVelWeight, n.bias, n.learningRate)
	}
}

// SetEpisode sets the current episode number
func (n *Network) SetEpisode(episode int) {
	n.currentEpisode = episode
	n.currentStep = 0
	
	// Update metrics logger if available
	if n.metrics != nil {
		n.metrics.SetEpisode(episode)
		// Log weights at the start of each episode to database
		n.metrics.LogWeights(n.angleWeight, n.angularVelWeight, n.bias, n.learningRate)
	} else if n.debug {
		// Only log to console if no metrics logger and debug is enabled
		n.logger.Printf("Starting episode %d with weights: angle=%.4f, angularVelWeight=%.4f, bias=%.4f, lr=%.4f",
			episode, n.angleWeight, n.angularVelWeight, n.bias, n.learningRate)
	}
}

// IncrementStep increments the current step counter
func (n *Network) IncrementStep() {
	n.currentStep++
	
	// Update metrics logger if available
	if n.metrics != nil {
		n.metrics.IncrementStep()
	}
}

// Forward performs a forward pass through the network
// Returns a force value in [-5, 5] Newtons
func (n *Network) Forward(state env.State) float64 {
	force, _ := n.ForwardWithActivation(state)
	return force
}

// ForwardWithActivation performs a forward pass and returns both the force and hidden layer activation
func (n *Network) ForwardWithActivation(state env.State) (float64, float64) {
	// Normalize angle to [-π, π] range
	angle := math.Mod(state.AngleRadians, 2*math.Pi)
	if angle > math.Pi {
		angle -= 2 * math.Pi
	} else if angle < -math.Pi {
		angle += 2 * math.Pi
	}
	
	// Get angular velocity
	velocity := state.AngularVel
	
	// Compute hidden activation
	// Negate angle and velocity to ensure correct force direction
	// When pendulum falls right (positive angle), we want negative force (push left)
	// When pendulum falls left (negative angle), we want positive force (push right)
	hidden := -n.angleWeight*angle - n.angularVelWeight*velocity + n.bias
	
	// Apply activation function (tanh)
	activation := math.Tanh(hidden)
	
	// Scale to force range [-5, 5] Newtons
	force := activation * 5.0
	
	// Store for learning
	n.lastForce = force
	n.lastInputs = []float64{angle, velocity}
	
	// Log metrics if available
	if n.metrics != nil {
		n.metrics.LogForwardPass(angle, velocity, force, hidden)
	} else if n.debug {
		// Only log to console if no metrics logger and debug is enabled
		n.logger.Printf("Forward: angle=%.4f, velocity=%.4f → force=%.4f", angle, velocity, force)
	}
	
	return force, hidden
}

// Predict estimates the value of a state for temporal difference learning
// Returns a value in [-1, 1] representing the estimated "goodness" of the state
func (n *Network) Predict(angleRadians, angularVel float64) float64 {
	// Normalize angle to [-π, π] range
	angle := math.Mod(angleRadians, 2*math.Pi)
	if angle > math.Pi {
		angle -= 2 * math.Pi
	} else if angle < -math.Pi {
		angle += 2 * math.Pi
	}
	
	// For prediction, we want to value states closer to balance (angle and velocity near zero)
	// So we use the negative of the absolute values
	balanceQuality := -math.Abs(angle) - 0.5*math.Abs(angularVel)
	
	// Apply activation function (tanh) to normalize to [-1, 1]
	n.lastValue = math.Tanh(balanceQuality)
	
	// Log prediction if metrics available
	if n.metrics != nil {
		n.metrics.LogPrediction(angle, angularVel, n.lastValue)
	} else if n.debug {
		// Only log to console if no metrics logger and debug is enabled
		n.logger.Printf("Predict: angle=%.4f, velocity=%.4f → value=%.4f", angle, angularVel, n.lastValue)
	}
	
	return n.lastValue
}

// Update adjusts weights based on the reward received
// reward should be in [-1, 1] range
func (n *Network) Update(reward float64) {
	// Ensure we have previous inputs
	if len(n.lastInputs) != 2 {
		return
	}

	// Track success/failure for progressive difficulty
	success := reward > 0.5
	n.updateSuccessRate(success)
	
	// Scale reward by difficulty for more aggressive learning at higher difficulties
	scaledReward := reward * (0.5 + 0.5*n.difficulty)
	
	// Compute error gradient
	error := scaledReward - n.lastValue
	
	// Apply learning rate (increased at higher difficulties)
	effectiveLR := n.learningRate * (1.0 + n.difficulty)
	
	// Update weights with momentum
	n.angleWeight += effectiveLR * error * n.lastInputs[0]
	n.angularVelWeight += effectiveLR * error * n.lastInputs[1]
	n.bias += effectiveLR * error
	
	// Log update if metrics available
	if n.metrics != nil {
		n.metrics.LogUpdate(error, n.angleWeight, n.angularVelWeight, n.bias, n.difficulty, n.successRate)
	} else if n.debug {
		n.logger.Printf("Update: reward=%.4f, error=%.4f, new_weights=[%.4f, %.4f, %.4f]",
			reward, error, n.angleWeight, n.angularVelWeight, n.bias)
	}
}

// updateSuccessRate updates the success tracking window and recalculates success rate
func (n *Network) updateSuccessRate(success bool) {
	// Add new result to window
	n.successWindow = append(n.successWindow, success)
	
	// Maintain window size
	if len(n.successWindow) > n.windowSize {
		n.successWindow = n.successWindow[1:]
	}
	
	// Calculate success rate
	successes := 0
	for _, s := range n.successWindow {
		if s {
			successes++
		}
	}
	n.successRate = float64(successes) / float64(len(n.successWindow))
	
	// Adjust difficulty based on success rate
	if len(n.successWindow) >= n.windowSize/2 { // Wait for sufficient data
		if n.successRate >= n.progressThresh {
			n.increaseDifficulty()
		} else if n.successRate <= n.regressThresh {
			n.decreaseDifficulty()
		}
	}
}

// increaseDifficulty increases the training difficulty
func (n *Network) increaseDifficulty() {
	oldDiff := n.difficulty
	n.difficulty = math.Min(1.0, n.difficulty*1.2) // 20% increase, capped at 1.0
	
	if n.metrics != nil {
		n.metrics.LogDifficultyChange(oldDiff, n.difficulty, "increase")
	} else if n.debug {
		n.logger.Printf("Increasing difficulty: %.4f → %.4f", oldDiff, n.difficulty)
	}
}

// decreaseDifficulty decreases the training difficulty
func (n *Network) decreaseDifficulty() {
	oldDiff := n.difficulty
	n.difficulty = math.Max(0.1, n.difficulty*0.8) // 20% decrease, minimum 0.1
	
	if n.metrics != nil {
		n.metrics.LogDifficultyChange(oldDiff, n.difficulty, "decrease")
	} else if n.debug {
		n.logger.Printf("Decreasing difficulty: %.4f → %.4f", oldDiff, n.difficulty)
	}
}

// GetDifficulty returns the current difficulty level
func (n *Network) GetDifficulty() float64 {
	return n.difficulty
}

// GetSuccessRate returns the current success rate
func (n *Network) GetSuccessRate() float64 {
	return n.successRate
}

// GetWeights returns the current network weights for testing
func (n *Network) GetWeights() []float64 {
	return []float64{n.angleWeight, n.angularVelWeight, n.bias}
}

// SetWeights updates the network weights with the provided values
// The weights slice must contain exactly 3 values: [angleWeight, angularVelWeight, bias]
func (n *Network) SetWeights(weights []float64) error {
	if len(weights) != 3 {
		return fmt.Errorf("expected 3 weights, got %d", len(weights))
	}
	
	n.angleWeight = weights[0]
	n.angularVelWeight = weights[1]
	n.bias = weights[2]
	
	// Record new weights in metrics if available
	if n.metrics != nil {
		n.metrics.LogWeights(n.angleWeight, n.angularVelWeight, n.bias, n.learningRate)
	} else if n.debug {
		n.logger.Printf("[Network] Set weights: angle=%.4f, angularVelWeight=%.4f, bias=%.4f\n",
			n.angleWeight, n.angularVelWeight, n.bias)
	}
	
	return nil
}

// SetLearningRate updates the learning rate
func (n *Network) SetLearningRate(rate float64) {
	oldRate := n.learningRate
	n.learningRate = rate
	
	// Log learning rate change to database if metrics available
	if n.metrics != nil {
		// Log the learning rate change as a weight update
		n.metrics.LogWeights(n.angleWeight, n.angularVelWeight, n.bias, rate)
		
		// Also log as a network operation for tracking system changes
		metadata := fmt.Sprintf("Changed learning rate from %.4f to %.4f", oldRate, rate)
		n.metrics.LogNetworkOperation("learning_rate_change", metadata, true)
	} else if n.debug {
		n.logger.Printf("[Network] Set learning rate: %.4f\n", rate)
	}
}

// GetLearningRate returns the current learning rate
func (n *Network) GetLearningRate() float64 {
	return n.learningRate
}

// sign returns the sign of a number: 1 for positive, -1 for negative, 0 for zero
func sign(x float64) float64 {
	if x > 0 {
		return 1.0
	} else if x < 0 {
		return -1.0
	}
	return 0.0
}

// clip limits a value to [min, max] range
func clip(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

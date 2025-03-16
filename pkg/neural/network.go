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
		angleWeight:      2.0,  // Strong initial response to angle
		angularVelWeight: 1.0,  // Moderate response to velocity
		bias:            0.0,  // Start with no bias
		learningRate:    0.05, // Learning rate for quick adaptation
		lastInputs:      make([]float64, 2),
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
	hidden := n.angleWeight*angle + n.angularVelWeight*velocity + n.bias
	
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
	
	// Compute hidden activation
	hidden := n.angleWeight*angle + n.angularVelWeight*angularVel + n.bias
	
	// Apply activation function (tanh)
	n.lastValue = math.Tanh(hidden)
	
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
	
	// Extract last inputs
	angle := n.lastInputs[0]
	angularVel := n.lastInputs[1]
	
	// Compute update values
	update := n.learningRate * reward
	
	// Compute weight updates
	angleUpdate := update * angle
	angularVelUpdate := update * angularVel
	biasUpdate := update
	
	// Apply updates
	n.angleWeight += angleUpdate
	n.angularVelWeight += angularVelUpdate
	n.bias += biasUpdate
	
	// Log updates if metrics available
	if n.metrics != nil {
		n.metrics.LogUpdate(reward, angleUpdate, angularVelUpdate, biasUpdate)
	} else if n.debug {
		// Only log to console if no metrics logger and debug is enabled
		n.logger.Printf("Update: reward=%.4f, updates=[%.4f, %.4f, %.4f]", 
			reward, angleUpdate, angularVelUpdate, biasUpdate)
	}
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

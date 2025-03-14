// Package neural implements a simple three-node neural network
// for controlling the inverted pendulum, following Pezzza's
// original implementation approach with minimal parameters
package neural

import (
	"fmt"
	"math"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
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

	// Debug flag
	debug bool
}

// NewNetwork creates a network with initialized weights
func NewNetwork() *Network {
	net := &Network{
		angleWeight:      2.0,  // Strong initial response to angle
		angularVelWeight: 1.0,  // Moderate response to velocity
		bias:            0.0,  // Start with no bias
		learningRate:    0.05, // Learning rate for quick adaptation
		lastInputs:      make([]float64, 2),
		debug:           true, // Enable debug by default
	}
	fmt.Printf("[Network] Created new network with initial weights: angle=%.4f, angularVel=%.4f, bias=%.4f\n",
		net.angleWeight, net.angularVelWeight, net.bias)
	return net
}

// SetDebug enables or disables debug printing
func (n *Network) SetDebug(enabled bool) {
	n.debug = enabled
}

// Forward performs a forward pass through the network
// Returns a force value in [-5, 5] Newtons
func (n *Network) Forward(state env.State) float64 {
	force, _ := n.ForwardWithActivation(state)
	return force
}

// ForwardWithActivation performs a forward pass and returns both the force and hidden layer activation
func (n *Network) ForwardWithActivation(state env.State) (float64, float64) {
	// Store inputs for weight updates
	n.lastInputs[0] = state.AngleRadians
	n.lastInputs[1] = state.AngularVel
	
	fmt.Printf("\n[Network Forward] State: angle=%.4f rad (%.1f°), angularVel=%.4f rad/s\n",
		state.AngleRadians,
		state.AngleRadians * 180 / math.Pi,
		state.AngularVel)
	fmt.Printf("[Network Forward] Current weights: angle=%.4f, angularVel=%.4f, bias=%.4f\n",
		n.angleWeight, n.angularVelWeight, n.bias)

	// Invert angle input so positive angle (falling right) generates negative force
	angle := -state.AngleRadians * 5.0  // Scale angle for stronger response
	angularVel := -state.AngularVel * 2.0  // Scale velocity for moderate response

	// Simple linear combination through hidden node
	hidden := n.angleWeight*angle +
		n.angularVelWeight*angularVel +
		n.bias
	
	fmt.Printf("[Network Forward] Scaled inputs: angle=%.4f, angularVel=%.4f\n", angle, angularVel)
	fmt.Printf("[Network Forward] Hidden activation: %.4f\n", hidden)
	
	// Hyperbolic tangent activation to bound output
	// Maps hidden value to [-1, 1], then scale to [-5, 5]
	n.lastForce = 5.0 * math.Tanh(hidden)
	
	fmt.Printf("[Network Forward] Output force: %.4f N\n", n.lastForce)

	return n.lastForce, hidden
}

// Update adjusts weights based on the reward received
// reward should be in [-1, 1] range
func (n *Network) Update(reward float64) {
	fmt.Printf("\n[Network Update] Received reward: %.4f\n", reward)
	fmt.Printf("[Network Update] Before weights: angle=%.4f, angularVel=%.4f, bias=%.4f\n", 
		n.angleWeight, n.angularVelWeight, n.bias)

	// Scale learning rate by reward magnitude
	update := n.learningRate * reward
	
	// Get the sign of the last force once for consistency
	forceSign := sign(n.lastForce)
	
	// Calculate weight updates
	angleUpdate := update * n.lastInputs[0] * forceSign
	angularVelUpdate := update * n.lastInputs[1] * forceSign
	biasUpdate := update * forceSign

	fmt.Printf("[Network Update] Updates: angle=%.4f, angularVel=%.4f, bias=%.4f\n",
		angleUpdate, angularVelUpdate, biasUpdate)
	
	// Update weights in direction that reinforces good actions
	// and weakens bad actions
	n.angleWeight += angleUpdate
	n.angularVelWeight += angularVelUpdate
	n.bias += biasUpdate
	
	// Optional: Clip weights to prevent explosion
	n.angleWeight = clip(n.angleWeight, -3.0, 3.0)     // Allow stronger angle response
	n.angularVelWeight = clip(n.angularVelWeight, -2.0, 2.0)  // Keep velocity response moderate
	n.bias = clip(n.bias, -1.0, 1.0)

	fmt.Printf("[Network Update] After weights: angle=%.4f, angularVel=%.4f, bias=%.4f\n", 
		n.angleWeight, n.angularVelWeight, n.bias)
}

// GetWeights returns the current network weights for testing
func (n *Network) GetWeights() []float64 {
	return []float64{n.angleWeight, n.angularVelWeight, n.bias}
}

// sign returns the sign of a number: 1 for positive, -1 for negative, 0 for zero
// This ensures deterministic behavior in our weight updates
func sign(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	if x < 0 {
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

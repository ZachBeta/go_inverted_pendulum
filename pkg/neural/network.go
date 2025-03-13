// Package neural implements a simple three-node neural network
// for controlling the inverted pendulum, following Pezzza's
// original implementation approach with minimal parameters
package neural

import (
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
}

// NewNetwork creates a network with initialized weights
func NewNetwork() *Network {
	return &Network{
		angleWeight:      0.1,  // Small initial weight for angle
		angularVelWeight: 0.1,  // Small initial weight for angular velocity
		bias:            0.0,  // Start with no bias
		learningRate:    0.01, // Small learning rate for stability
		lastInputs:      make([]float64, 2),
	}
}

// Forward performs a forward pass through the network
// Returns a force value in [-5, 5] Newtons
func (n *Network) Forward(state env.State) float64 {
	// Store inputs for weight updates
	n.lastInputs[0] = state.AngleRadians
	n.lastInputs[1] = state.AngularVel
	
	// Simple linear combination through hidden node
	hidden := n.angleWeight*state.AngleRadians +
		n.angularVelWeight*state.AngularVel +
		n.bias
	
	// Hyperbolic tangent activation to bound output
	// Maps hidden value to [-1, 1], then scale to [-5, 5]
	n.lastForce = 5.0 * math.Tanh(hidden)
	
	return n.lastForce
}

// Update adjusts weights based on the reward received
// reward should be in [-1, 1] range
func (n *Network) Update(reward float64) {
	// Scale learning rate by reward magnitude
	update := n.learningRate * reward
	
	// Get the sign of the last force once for consistency
	forceSign := sign(n.lastForce)
	
	// Update weights in direction that reinforces good actions
	// and weakens bad actions
	n.angleWeight += update * n.lastInputs[0] * forceSign
	n.angularVelWeight += update * n.lastInputs[1] * forceSign
	n.bias += update * forceSign
	
	// Optional: Clip weights to prevent explosion
	n.angleWeight = clip(n.angleWeight, -1.0, 1.0)
	n.angularVelWeight = clip(n.angularVelWeight, -1.0, 1.0)
	n.bias = clip(n.bias, -1.0, 1.0)
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

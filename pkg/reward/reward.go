// Package reward provides reward calculation for the inverted pendulum system
package reward

import (
	"math"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
)

// RewardCalculator provides a simple angle-based reward system
// following the progressive complexity approach
type RewardCalculator struct{}

// NewRewardCalculator creates a new reward calculator
func NewRewardCalculator() *RewardCalculator {
	return &RewardCalculator{}
}

// Calculate computes a simple reward based solely on pendulum angle
// Returns a value in [-1, 1] where:
// 1.0 = perfectly upright (0 radians)
// 0.0 = horizontal (±π/2 radians)
// -1.0 = hanging down (±π radians)
func (r *RewardCalculator) Calculate(state env.State) float64 {
	// Normalize angle to [-π, π] for reward calculation
	angle := normalizeAngle(state.AngleRadians)
	
	// Use cosine function to map angle to reward:
	// cos(0) = 1.0 (upright)
	// cos(±π/2) = 0.0 (horizontal)
	// cos(±π) = -1.0 (hanging)
	return math.Cos(angle)
}

// normalizeAngle converts any angle to [-π, π] range
func normalizeAngle(angle float64) float64 {
	// First normalize to [0, 2π)
	angle = math.Mod(angle, 2*math.Pi)
	if angle < 0 {
		angle += 2 * math.Pi
	}
	
	// Then convert to [-π, π]
	if angle > math.Pi {
		angle -= 2 * math.Pi
	}
	
	return angle
}

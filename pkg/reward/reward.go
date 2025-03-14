// Package reward provides reward calculation for the inverted pendulum system
package reward

import (
	"fmt"
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

// Calculate computes a reward based on the change in pendulum state
// Returns a value in [-1, 1] where:
// 1.0 = perfect improvement (moving towards upright)
// 0.0 = no change
// -1.0 = worst deterioration (moving towards hanging)
func Calculate(prevState, newState env.State) float64 {
	fmt.Printf("\n[Reward] Previous state: angle=%.4f rad (%.1f°), pos=%.4f m\n",
		prevState.AngleRadians,
		prevState.AngleRadians * 180 / math.Pi,
		prevState.CartPosition)
	fmt.Printf("[Reward] New state: angle=%.4f rad (%.1f°), pos=%.4f m\n",
		newState.AngleRadians,
		newState.AngleRadians * 180 / math.Pi,
		newState.CartPosition)

	// Get angle-based rewards for both states
	calc := NewRewardCalculator()
	prevReward := calc.Calculate(prevState)
	newReward := calc.Calculate(newState)

	fmt.Printf("[Reward] Angle-based rewards: prev=%.4f, new=%.4f\n", prevReward, newReward)

	// Calculate improvement (can be negative)
	improvement := newReward - prevReward

	// Add small penalty for cart position to keep it centered
	positionPenalty := math.Abs(newState.CartPosition) * 0.1

	// Add larger penalty if near track bounds
	boundsPenalty := 0.0
	if math.Abs(newState.CartPosition) > 1.5 {
		boundsPenalty = 0.5 // Strong penalty when getting close to bounds
	}

	fmt.Printf("[Reward] Improvement=%.4f, PositionPenalty=%.4f, BoundsPenalty=%.4f\n",
		improvement, positionPenalty, boundsPenalty)

	// Combine rewards, ensuring output is in [-1, 1]
	finalReward := clip(improvement - positionPenalty - boundsPenalty, -1.0, 1.0)
	fmt.Printf("[Reward] Final reward: %.4f\n", finalReward)

	return finalReward
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

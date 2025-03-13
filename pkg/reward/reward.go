// Package reward provides reward calculation for the inverted pendulum system
package reward

import (
	"math"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
)

// RewardCalculator handles both immediate and delayed reward calculations
// using deterministic, performance-optimized methods
type RewardCalculator struct {
	// Configurable weights for different components of the reward
	angleWeight     float64
	positionWeight  float64
	velocityWeight  float64
	stabilityWeight float64
}

// NewRewardCalculator creates a new reward calculator with default weights
func NewRewardCalculator() *RewardCalculator {
	return &RewardCalculator{
		angleWeight:     0.6,  // Primary weight for angle
		positionWeight:  0.2,  // Secondary weight for position
		velocityWeight:  0.2,  // Secondary weight for velocities
		stabilityWeight: 1.0,  // Base weight for stability
	}
}

// CalculateImmediate computes the immediate reward for a given state
// Returns a value in [-1, 1] where:
// 1.0 = perfect upright position with minimal velocities
// -1.0 = hanging down or extreme instability
func (r *RewardCalculator) CalculateImmediate(state env.State) float64 {
	// Normalize angle to [-π, π] for reward calculation
	angle := normalizeAngleForReward(state.AngleRadians)
	
	// Angle component: 1.0 when upright (0), -1.0 when hanging (±π)
	angleReward := math.Cos(angle)
	
	// Position component: quadratic penalty for distance from center
	positionPenalty := math.Pow(state.CartPosition/2.0, 2)
	positionReward := math.Max(-1.0, 1.0-positionPenalty)
	
	// Velocity component: combined cart and angular velocity penalties
	velocityPenalty := (math.Abs(state.CartVelocity) + 
		2.0*math.Abs(state.AngularVel)) / 4.0
	velocityReward := math.Max(-1.0, 1.0-velocityPenalty)
	
	// Calculate total reward
	reward := angleReward
	
	// Apply position and velocity penalties
	if positionReward < 1.0 {
		reward *= (1.0 + positionReward) / 2.0
	}
	if velocityReward < 1.0 {
		reward *= (1.0 + velocityReward) / 2.0
	}
	
	return reward
}

// CalculateDelayed computes the delayed reward based on a sequence of states
// Returns a value in [-2, 2] where:
// 2.0 = excellent stabilization progress
// -2.0 = increasing instability or loss of control
func (r *RewardCalculator) CalculateDelayed(states []env.State) float64 {
	if len(states) < 2 {
		return 0.0
	}
	
	// Track angle deviations and their progression
	initialDev := math.Abs(normalizeAngleForReward(states[0].AngleRadians))
	maxDev := initialDev
	var stabilityScore float64
	
	// Analyze progression through states
	for i := 1; i < len(states); i++ {
		currentDev := math.Abs(normalizeAngleForReward(states[i].AngleRadians))
		
		// Calculate improvement relative to worst case
		if maxDev > 0 {
			improvement := (maxDev - currentDev) / maxDev
			if improvement > 0 {
				stabilityScore += improvement
			} else {
				// Penalize getting worse more heavily
				stabilityScore += 2.0 * improvement
			}
		}
		
		// Update maximum deviation
		maxDev = math.Max(maxDev, currentDev)
	}
	
	// Calculate time-weighted average of immediate rewards
	var avgImmediate float64
	var totalWeight float64
	
	for i, state := range states {
		weight := float64(i + 1) // Recent states matter more
		reward := r.CalculateImmediate(state)
		avgImmediate += weight * reward
		totalWeight += weight
	}
	avgImmediate /= totalWeight
	
	// Scale stability score by number of transitions
	stabilityScore = (stabilityScore / float64(len(states)-1))
	
	// Combine immediate and stability components
	var finalReward float64
	if avgImmediate >= 0 {
		// For positive performance, amplify stability improvements
		finalReward = avgImmediate + stabilityScore
	} else {
		// For negative performance, emphasize instability
		finalReward = avgImmediate - math.Abs(stabilityScore)
	}
	
	// Scale to target ranges:
	// [-1.0, 1.5] for recovery and stability
	// [-2.0, -1.0] for increasing instability
	if finalReward >= 0 {
		return 1.5 * finalReward
	} else {
		return -1.0 - math.Abs(finalReward)
	}
}

// normalizeAngleForReward converts any angle to [-π, π] range
// This is optimized for reward calculation where we care about
// deviation from upright position (0) in either direction
func normalizeAngleForReward(angle float64) float64 {
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

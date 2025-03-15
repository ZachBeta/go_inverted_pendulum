package neural

import (
	"math"
	"testing"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
)

func TestThreeNodeNetwork(t *testing.T) {
	tests := []struct {
		name        string
		state       env.State
		wantForce   float64
		tolerance   float64
		description string
	}{
		{
			name: "zero_state",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: 0,
				AngularVel:   0,
			},
			wantForce:   0,
			tolerance:   0.1,
			description: "Network should output no force when pendulum is perfectly balanced",
		},
		{
			name: "falling_right",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: 0.1, // slight tilt right
				AngularVel:   0.2, // falling right
			},
			wantForce:   -4.5, // Progressive force for small angle
			tolerance:   0.5,
			description: "Network should push left proportionally when pendulum falls right",
		},
		{
			name: "falling_left",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: -0.1, // slight tilt left
				AngularVel:   -0.2, // falling left
			},
			wantForce:   4.5, // Progressive force for small angle
			tolerance:   0.5,
			description: "Network should push right proportionally when pendulum falls left",
		},
		{
			name: "extreme_right",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: 0.5, // significant tilt right
				AngularVel:   1.0, // falling fast right
			},
			wantForce:   -5.0, // Maximum force for extreme angle
			tolerance:   0.1,
			description: "Network should apply maximum force for extreme angles",
		},
		{
			name: "extreme_left",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: -0.5, // significant tilt left
				AngularVel:   -1.0, // falling fast left
			},
			wantForce:   5.0, // Maximum force for extreme angle
			tolerance:   0.1,
			description: "Network should apply maximum force for extreme angles",
		},
	}

	network := NewNetwork()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := network.Forward(tt.state)
			if math.Abs(got-tt.wantForce) > tt.tolerance {
				t.Errorf("%s: got force %.4f, want %.4f (±%.4f)", 
					tt.description, got, tt.wantForce, tt.tolerance)
			}
		})
	}
}

func TestWeightUpdates(t *testing.T) {
	tests := []struct {
		name        string
		state       env.State
		reward      float64
		description string
	}{
		{
			name: "positive_reinforcement",
			state: env.State{
				AngleRadians: 0.1,
				AngularVel:   0.2,
			},
			reward:      0.8,
			description: "Weights should strengthen when action leads to good outcome",
		},
		{
			name: "negative_reinforcement",
			state: env.State{
				AngleRadians: 0.2,
				AngularVel:   0.3,
			},
			reward:      -0.5,
			description: "Weights should weaken when action leads to poor outcome",
		},
	}

	network := NewNetwork()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Record initial weights
			initialWeights := network.GetWeights()
			
			// Forward pass and update
			force := network.Forward(tt.state)
			network.Update(tt.reward)
			
			// Get updated weights
			newWeights := network.GetWeights()
			
			// Check weight changes align with reward
			if tt.reward > 0 {
				// For positive reward, weights should move to strengthen the output
				if force > 0 && !weightsIncreased(initialWeights, newWeights) {
					t.Errorf("%s: weights did not strengthen for positive reward", tt.description)
				}
				if force < 0 && !weightsDecreased(initialWeights, newWeights) {
					t.Errorf("%s: weights did not strengthen for positive reward", tt.description)
				}
			} else {
				// For negative reward, weights should move to weaken the output
				if force > 0 && !weightsDecreased(initialWeights, newWeights) {
					t.Errorf("%s: weights did not weaken for negative reward", tt.description)
				}
				if force < 0 && !weightsIncreased(initialWeights, newWeights) {
					t.Errorf("%s: weights did not weaken for negative reward", tt.description)
				}
			}
		})
	}
}

func TestLearningBehavior(t *testing.T) {
	scenarios := []struct {
		name        string
		episodes    []struct {
			state  env.State
			reward float64
		}
		description string
	}{
		{
			name: "learn_to_balance",
			episodes: []struct {
				state  env.State
				reward float64
			}{
				{
					state: env.State{AngleRadians: 0.1, AngularVel: 0.2},
					reward: -0.5, // Penalize for being off-center
				},
				{
					state: env.State{AngleRadians: 0.05, AngularVel: 0.1},
					reward: -0.2, // Less penalty as we get closer
				},
				{
					state: env.State{AngleRadians: 0.01, AngularVel: 0.02},
					reward: 0.8, // Reward for being close to center
				},
			},
			description: "Network should learn to keep pendulum centered",
		},
		{
			name: "recover_from_fall",
			episodes: []struct {
				state  env.State
				reward float64
			}{
				{
					state: env.State{AngleRadians: 0.5, AngularVel: 1.0},
					reward: -0.8, // Heavy penalty for extreme angle
				},
				{
					state: env.State{AngleRadians: 0.3, AngularVel: 0.5},
					reward: -0.3, // Less penalty as angle reduces
				},
				{
					state: env.State{AngleRadians: 0.1, AngularVel: 0.2},
					reward: 0.5, // Reward for recovering
				},
			},
			description: "Network should learn recovery behavior",
		},
	}

	network := NewNetwork()
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			var lastForce float64
			var improvedCount int

			for i, episode := range scenario.episodes {
				force := network.Forward(episode.state)
				network.Update(episode.reward)

				if i > 0 {
					// Check if force application is improving
					if math.Abs(force) < math.Abs(lastForce) {
						improvedCount++
					}
				}
				lastForce = force
			}

			// Expect improvement in at least half of the episodes
			if improvedCount < len(scenario.episodes)/2 {
				t.Errorf("%s: network did not show consistent improvement", scenario.description)
			}
		})
	}
}

func TestTemporalDifferenceLearning(t *testing.T) {
	tests := []struct {
		name          string
		currentState  env.State
		nextState     env.State
		wantHigherVal bool // true if next state should be valued higher
		description   string
	}{
		{
			name: "value_improvement",
			currentState: env.State{
				AngleRadians: 0.3,  // Significant tilt
				AngularVel:   0.5,  // Moving away from center
			},
			nextState: env.State{
				AngleRadians: 0.1,  // Less tilt
				AngularVel:   -0.2, // Moving toward center
			},
			wantHigherVal: true,
			description:  "State moving toward balance should have higher value",
		},
		{
			name: "value_deterioration",
			currentState: env.State{
				AngleRadians: 0.1,  // Small tilt
				AngularVel:   0.1,  // Slow movement
			},
			nextState: env.State{
				AngleRadians: 0.3,  // Larger tilt
				AngularVel:   0.6,  // Faster movement away
			},
			wantHigherVal: false,
			description:  "State moving away from balance should have lower value",
		},
		{
			name: "stable_state",
			currentState: env.State{
				AngleRadians: 0.05, // Very small tilt
				AngularVel:   0.1,  // Slow movement
			},
			nextState: env.State{
				AngleRadians: 0.05, // Same tilt
				AngularVel:   0.1,  // Same movement
			},
			wantHigherVal: false,
			description:  "Similar states should have similar values",
		},
	}

	network := NewNetwork()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			currentVal := network.Predict(tt.currentState.AngleRadians, tt.currentState.AngularVel)
			nextVal := network.Predict(tt.nextState.AngleRadians, tt.nextState.AngularVel)

			if tt.wantHigherVal && nextVal <= currentVal {
				t.Errorf("%s: next state (%.4f) should be valued higher than current state (%.4f)",
					tt.description, nextVal, currentVal)
			} else if !tt.wantHigherVal && nextVal > currentVal+0.1 { // Allow small increase
				t.Errorf("%s: next state (%.4f) should not be valued higher than current state (%.4f)",
					tt.description, nextVal, currentVal)
			}
		})
	}
}

func BenchmarkNetwork(b *testing.B) {
	network := NewNetwork()
	network.SetDebug(false) // Disable debug output for benchmarking
	state := env.State{
		CartPosition: 0,
		CartVelocity: 0,
		AngleRadians: 0.1,
		AngularVel:   0.2,
	}

	b.Run("Forward", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			network.Forward(state)
		}
	})

	b.Run("Predict", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			network.Predict(state.AngleRadians, state.AngularVel)
		}
	})

	b.Run("Update", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			network.Update(0.5)
		}
	})

	b.Run("FullStep", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			network.Forward(state)
			network.Predict(state.AngleRadians, state.AngularVel)
			network.Update(0.5)
		}
	})
}

// Helper functions for comparing weights
func weightsIncreased(old, new []float64) bool {
	increased := false
	for i := range old {
		if new[i] > old[i] {
			increased = true
		}
	}
	return increased
}

func weightsDecreased(old, new []float64) bool {
	decreased := false
	for i := range old {
		if new[i] < old[i] {
			decreased = true
		}
	}
	return decreased
}

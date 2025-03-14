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
			description: "Network should learn to recover from falling",
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			network := NewNetwork()
			network.SetDebug(true) // Enable debug prints for learning analysis
			
			// Run through learning episodes
			var forces []float64
			for _, episode := range scenario.episodes {
				force := network.Forward(episode.state)
				forces = append(forces, force)
				network.Update(episode.reward)
			}
			
			// Verify learning progress
			if len(forces) < 2 {
				t.Fatal("Need at least 2 episodes to verify learning")
			}
			
			// For balancing scenario, forces should get smaller
			if scenario.name == "learn_to_balance" {
				if math.Abs(forces[len(forces)-1]) >= math.Abs(forces[0]) {
					t.Errorf("%s: forces did not decrease over time. Initial: %.4f, Final: %.4f",
						scenario.description, forces[0], forces[len(forces)-1])
				}
			}
			
			// For recovery scenario, forces should align with fall direction
			if scenario.name == "recover_from_fall" {
				lastForce := forces[len(forces)-1]
				lastState := scenario.episodes[len(scenario.episodes)-1].state
				if lastState.AngleRadians > 0 && lastForce > 0 {
					t.Errorf("%s: network pushed right when pendulum tilted right", scenario.description)
				}
				if lastState.AngleRadians < 0 && lastForce < 0 {
					t.Errorf("%s: network pushed left when pendulum tilted left", scenario.description)
				}
			}
		})
	}
}

func BenchmarkNetwork(b *testing.B) {
	network := NewNetwork()
	state := env.State{
		CartPosition: 0.1,
		CartVelocity: 0.2,
		AngleRadians: 0.3,
		AngularVel:   0.4,
	}

	b.Run("forward_pass", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			network.Forward(state)
		}
	})

	b.Run("weight_update", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			network.Update(0.5)
		}
	})
}

// Helper functions for comparing weights
func weightsIncreased(old, new []float64) bool {
	if len(old) != len(new) {
		return false
	}
	increased := false
	for i := range old {
		if new[i] < old[i] {
			return false
		}
		if new[i] > old[i] {
			increased = true
		}
	}
	return increased
}

func weightsDecreased(old, new []float64) bool {
	if len(old) != len(new) {
		return false
	}
	decreased := false
	for i := range old {
		if new[i] > old[i] {
			return false
		}
		if new[i] < old[i] {
			decreased = true
		}
	}
	return decreased
}

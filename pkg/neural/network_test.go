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
			wantForce:   -5.0, // should push left to counteract
			description: "Network should push left when pendulum falls right",
		},
		{
			name: "falling_left",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: -0.1, // slight tilt left
				AngularVel:   -0.2, // falling left
			},
			wantForce:   5.0, // should push right to counteract
			description: "Network should push right when pendulum falls left",
		},
	}

	network := NewNetwork()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := network.Forward(tt.state)
			if math.Abs(got-tt.wantForce) > 1e-6 {
				t.Errorf("%s: got force %.6f, want %.6f", tt.description, got, tt.wantForce)
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

package reward

import (
	"math"
	"testing"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
)

func TestImmediateReward(t *testing.T) {
	tests := []struct {
		name        string
		state       env.State
		wantReward  float64
		description string
	}{
		{
			name: "perfect_balance",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: 0, // upright
				AngularVel:   0,
				TimeStep:     0,
			},
			wantReward:  1.0,
			description: "Maximum reward when pendulum is perfectly balanced upright",
		},
		{
			name: "hanging_down",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: math.Pi, // hanging down
				AngularVel:   0,
				TimeStep:     0,
			},
			wantReward:  -1.0,
			description: "Minimum reward when pendulum is hanging down",
		},
		{
			name: "horizontal_position",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 0,
				AngleRadians: math.Pi / 2, // horizontal
				AngularVel:   0,
				TimeStep:     0,
			},
			wantReward:  0.0,
			description: "Zero reward when pendulum is horizontal",
		},
		{
			name: "cart_far_from_center",
			state: env.State{
				CartPosition: 0.5, // half meter from center
				CartVelocity: 0,
				AngleRadians: 0,
				AngularVel:   0,
				TimeStep:     0,
			},
			wantReward:  0.5, // reduced reward due to cart position
			description: "Reduced reward when cart is away from center",
		},
		{
			name: "high_velocities",
			state: env.State{
				CartPosition: 0,
				CartVelocity: 2.0,
				AngleRadians: 0,
				AngularVel:   1.0,
				TimeStep:     0,
			},
			wantReward:  0.3, // reduced reward due to high velocities
			description: "Reduced reward when velocities are high",
		},
	}

	calculator := NewRewardCalculator()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculator.CalculateImmediate(tt.state)
			if math.Abs(got-tt.wantReward) > 1e-6 {
				t.Errorf("%s: got reward %.6f, want %.6f", tt.description, got, tt.wantReward)
			}
		})
	}
}

func TestDelayedReward(t *testing.T) {
	tests := []struct {
		name        string
		states      []env.State
		wantReward  float64
		description string
	}{
		{
			name: "stable_upright",
			states: []env.State{
				{AngleRadians: 0.1, TimeStep: 0},
				{AngleRadians: 0.05, TimeStep: 1},
				{AngleRadians: 0.02, TimeStep: 2},
				{AngleRadians: 0.01, TimeStep: 3},
			},
			wantReward:  2.0,
			description: "High reward for stabilizing towards upright",
		},
		{
			name: "unstable_movement",
			states: []env.State{
				{AngleRadians: 0.1, TimeStep: 0},
				{AngleRadians: 0.2, TimeStep: 1},
				{AngleRadians: 0.3, TimeStep: 2},
				{AngleRadians: 0.4, TimeStep: 3},
			},
			wantReward:  -1.0,
			description: "Negative reward for increasing instability",
		},
		{
			name: "recovery_from_tilt",
			states: []env.State{
				{AngleRadians: 0.5, TimeStep: 0},
				{AngleRadians: 0.4, TimeStep: 1},
				{AngleRadians: 0.2, TimeStep: 2},
				{AngleRadians: 0.1, TimeStep: 3},
			},
			wantReward:  1.5,
			description: "Positive reward for recovery towards upright",
		},
	}

	calculator := NewRewardCalculator()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculator.CalculateDelayed(tt.states)
			if math.Abs(got-tt.wantReward) > 1e-6 {
				t.Errorf("%s: got reward %.6f, want %.6f", tt.description, got, tt.wantReward)
			}
		})
	}
}

func BenchmarkRewardCalculation(b *testing.B) {
	calculator := NewRewardCalculator()
	state := env.State{
		CartPosition: 0.1,
		CartVelocity: 0.2,
		AngleRadians: 0.3,
		AngularVel:   0.4,
		TimeStep:     0,
	}

	b.Run("immediate_reward", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			calculator.CalculateImmediate(state)
		}
	})

	states := make([]env.State, 100)
	for i := range states {
		states[i] = env.State{
			CartPosition: float64(i) * 0.01,
			AngleRadians: float64(i) * 0.01,
			TimeStep:     uint64(i),
		}
	}

	b.Run("delayed_reward", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			calculator.CalculateDelayed(states)
		}
	})
}

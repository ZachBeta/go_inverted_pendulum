package reward

import (
	"math"
	"testing"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
)

func TestBasicReward(t *testing.T) {
	tests := []struct {
		name        string
		state       env.State
		wantReward  float64
		description string
	}{
		{
			name: "upright",
			state: env.State{
				AngleRadians: 0, // upright
			},
			wantReward:  1.0,
			description: "Maximum reward when pendulum is upright",
		},
		{
			name: "hanging_down",
			state: env.State{
				AngleRadians: math.Pi, // hanging down
			},
			wantReward:  -1.0,
			description: "Minimum reward when pendulum is hanging down",
		},
		{
			name: "horizontal_right",
			state: env.State{
				AngleRadians: math.Pi / 2, // horizontal
			},
			wantReward:  0.0,
			description: "Zero reward when pendulum is horizontal (right)",
		},
		{
			name: "horizontal_left",
			state: env.State{
				AngleRadians: -math.Pi / 2, // horizontal
			},
			wantReward:  0.0,
			description: "Zero reward when pendulum is horizontal (left)",
		},
	}

	calculator := NewRewardCalculator()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculator.Calculate(tt.state)
			if math.Abs(got-tt.wantReward) > 1e-6 {
				t.Errorf("%s: got reward %.6f, want %.6f", tt.description, got, tt.wantReward)
			}
		})
	}
}

func BenchmarkBasicReward(b *testing.B) {
	calculator := NewRewardCalculator()
	state := env.State{
		AngleRadians: 0.3,
	}

	b.Run("reward_calculation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			calculator.Calculate(state)
		}
	})
}

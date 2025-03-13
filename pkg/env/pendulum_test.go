package env

import (
	"bytes"
	"log"
	"math"
	"strings"
	"testing"
)

func TestNewPendulum(t *testing.T) {
	config := NewDefaultConfig()
	p := NewPendulum(config, nil)

	if p == nil {
		t.Fatal("NewPendulum returned nil")
	}

	state := p.GetState()
	if state.AngleRadians != math.Pi {
		t.Errorf("Expected initial angle %.2f, got %.2f", math.Pi, state.AngleRadians)
	}
}

func TestPendulumStep(t *testing.T) {
	tests := []struct {
		name      string
		force     float64
		steps     int
		wantError bool
	}{
		{
			name:      "Zero force",
			force:     0,
			steps:     10,
			wantError: false,
		},
		{
			name:      "Positive force within bounds",
			force:     5.0,
			steps:     10,
			wantError: false,
		},
		{
			name:      "Force exceeding max",
			force:     20.0, // Should be clamped to MaxForce
			steps:     10,
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var logBuf bytes.Buffer
			logger := log.New(&logBuf, "", 0)
			
			p := NewPendulum(NewDefaultConfig(), logger)
			var lastState State
			var err error

			for i := 0; i < tt.steps; i++ {
				lastState, err = p.Step(tt.force)
				if (err != nil) != tt.wantError {
					t.Errorf("Step() error = %v, wantError %v", err, tt.wantError)
					return
				}
			}

			// Verify state changes
			if lastState.TimeStep != uint64(tt.steps) {
				t.Errorf("Expected %d steps, got %d", tt.steps, lastState.TimeStep)
			}

			// Verify logging
			logOutput := logBuf.String()
			if !strings.Contains(logOutput, "Applying force") {
				t.Error("Expected force application to be logged")
			}
			if !strings.Contains(logOutput, "New state") {
				t.Error("Expected state updates to be logged")
			}
		})
	}
}

func TestNormalizeAngle(t *testing.T) {
	tests := []struct {
		desc     string
		angle    float64
		expected float64
	}{
		// Basic range [0, 2π]
		{
			desc:     "zero remains zero",
			angle:    0,
			expected: 0,
		},
		{
			desc:     "π/4 remains π/4",
			angle:    math.Pi / 4,
			expected: math.Pi / 4,
		},
		{
			desc:     "π/2 remains π/2",
			angle:    math.Pi / 2,
			expected: math.Pi / 2,
		},
		{
			desc:     "3π/4 remains 3π/4",
			angle:    3 * math.Pi / 4,
			expected: 3 * math.Pi / 4,
		},
		{
			desc:     "π remains π",
			angle:    math.Pi,
			expected: math.Pi,
		},
		{
			desc:     "5π/4 remains 5π/4",
			angle:    5 * math.Pi / 4,
			expected: 5 * math.Pi / 4,
		},
		{
			desc:     "3π/2 remains 3π/2",
			angle:    3 * math.Pi / 2,
			expected: 3 * math.Pi / 2,
		},
		{
			desc:     "7π/4 remains 7π/4",
			angle:    7 * math.Pi / 4,
			expected: 7 * math.Pi / 4,
		},
		{
			desc:     "2π wraps to 0",
			angle:    2 * math.Pi,
			expected: 0,
		},
		// Beyond 2π
		{
			desc:     "5π/2 wraps to π/2",
			angle:    5 * math.Pi / 2,
			expected: math.Pi / 2,
		},
		{
			desc:     "3π wraps to π",
			angle:    3 * math.Pi,
			expected: math.Pi,
		},
		{
			desc:     "slightly over 3π wraps to slightly over π",
			angle:    9.42, // ~3π + 0.28
			expected: 3.1368146928204137,
		},
		// Negative angles
		{
			desc:     "-π/2 wraps to 3π/2",
			angle:    -math.Pi / 2,
			expected: 3 * math.Pi / 2,
		},
		{
			desc:     "-π wraps to π",
			angle:    -math.Pi,
			expected: math.Pi,
		},
		{
			desc:     "-3π/2 wraps to π/2",
			angle:    -3 * math.Pi / 2,
			expected: math.Pi / 2,
		},
		{
			desc:     "-2π wraps to 0",
			angle:    -2 * math.Pi,
			expected: 0,
		},
		{
			desc:     "-3π wraps to π",
			angle:    -3 * math.Pi,
			expected: math.Pi,
		},
		{
			desc:     "slightly under -3π wraps to slightly over π",
			angle:    -9.42, // ~-3π - 0.28
			expected: 3.1368146928204137,
		},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			result := NormalizeAngle(tt.angle)
			if math.Abs(result-tt.expected) > 1e-2 {
				t.Errorf("NormalizeAngle(%v) = %v; want %v", 
					tt.angle, result, tt.expected)
			}
		})
	}
}

func TestTrackBounds(t *testing.T) {
	config := NewDefaultConfig()
	config.TrackLength = 2.0 // Small track to test bounds

	p := NewPendulum(config, nil)
	
	// Apply large force to push cart towards bounds
	force := config.MaxForce
	var err error
	
	// Keep stepping until we hit bounds or timeout
	for i := 0; i < 100; i++ {
		_, err = p.Step(force)
		if err != nil {
			if !strings.Contains(err.Error(), "exceeds track bounds") {
				t.Errorf("Expected track bounds error, got: %v", err)
			}
			return
		}
	}
	
	t.Error("Expected to hit track bounds but didn't")
}

func TestDeterministicReplay(t *testing.T) {
	config := NewDefaultConfig()
	forces := []float64{2.0, -3.0, 4.0, -2.0, 1.0}
	
	// First run
	p1 := NewPendulum(config, nil)
	var states1 []State
	for _, force := range forces {
		state, err := p1.Step(force)
		if err != nil {
			t.Fatalf("First run failed: %v", err)
		}
		states1 = append(states1, state)
	}

	// Second run with same inputs
	p2 := NewPendulum(config, nil)
	var states2 []State
	for _, force := range forces {
		state, err := p2.Step(force)
		if err != nil {
			t.Fatalf("Second run failed: %v", err)
		}
		states2 = append(states2, state)
	}

	// Compare states
	for i := range states1 {
		if states1[i] != states2[i] {
			t.Errorf("Non-deterministic behavior at step %d:\nFirst run: %+v\nSecond run: %+v",
				i, states1[i], states2[i])
		}
	}
}

func TestStateTransitions(t *testing.T) {
	config := NewDefaultConfig()
	p := NewPendulum(config, nil)
	
	// Test state transitions with constant force
	force := 3.0
	var prevState State
	var workDone float64 // Track work done by external force
	
	for i := 0; i < 10; i++ {
		state, err := p.Step(force)
		if err != nil {
			t.Fatalf("Step %d failed: %v", i, err)
		}

		if i > 0 {
			// Verify time step increment
			if state.TimeStep != prevState.TimeStep+1 {
				t.Errorf("Step %d: TimeStep not incremented correctly", i)
			}

			// Verify cart movement direction matches force
			if force > 0 && state.CartVelocity <= prevState.CartVelocity {
				t.Errorf("Step %d: Cart velocity not increasing with positive force", i)
			}

			// Calculate work done by force (F * dx)
			dx := state.CartPosition - prevState.CartPosition
			workDone += force * dx

			// Verify energy conservation including work done
			prevEnergy := calculateSystemEnergy(prevState)
			currentEnergy := calculateSystemEnergy(state)
			energyDiff := math.Abs((currentEnergy - prevEnergy) - (force * dx))
			if energyDiff > 1e-2 {
				t.Errorf("Step %d: Energy not conserved (accounting for work), diff: %v", i, energyDiff)
			}
		}
		
		prevState = state
	}
}

func calculateSystemEnergy(s State) float64 {
	config := NewDefaultConfig()
	// Kinetic + Potential energy
	cartKE := 0.5 * config.CartMass * s.CartVelocity * s.CartVelocity
	pendulumKE := 0.5 * config.PendulumMass * s.AngularVel * s.AngularVel
	potentialE := config.PendulumMass * config.Gravity * (1 - math.Cos(s.AngleRadians))
	return cartKE + pendulumKE + potentialE
}

func BenchmarkPendulumStep(b *testing.B) {
	config := NewDefaultConfig()
	p := NewPendulum(config, nil)
	force := 2.0

	b.Run("single_step", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = p.Step(force)
		}
	})

	b.Run("state_transitions", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := NewPendulum(config, nil)
			for j := 0; j < 100; j++ {
				_, _ = p.Step(force)
			}
		}
	})
}

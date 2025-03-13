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

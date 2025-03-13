package env

import "math"

// State represents the immutable state of the pendulum system
type State struct {
	CartPosition  float64 // x position of cart
	CartVelocity  float64 // velocity of cart
	AngleRadians  float64 // angle of pendulum in radians (0 is upright)
	AngularVel    float64 // angular velocity of pendulum
	TimeStep      uint64  // current simulation timestep
}

// Config holds the physical parameters of the system
type Config struct {
	CartMass     float64 // mass of the cart in kg
	PendulumMass float64 // mass of the pendulum in kg
	Length       float64 // length of pendulum in meters
	Gravity      float64 // gravitational acceleration in m/s²
	MaxForce     float64 // maximum force that can be applied to cart
	DeltaTime    float64 // simulation timestep in seconds
	TrackLength  float64 // length of the track in meters
}

// NewDefaultConfig returns a Config with reasonable default values
func NewDefaultConfig() Config {
	return Config{
		CartMass:     1.0,
		PendulumMass: 0.1,
		Length:       1.0,
		Gravity:      9.81,
		MaxForce:     10.0,
		DeltaTime:    0.02,
		TrackLength:  4.0,
	}
}

// NormalizeAngle ensures angle stays within [0, 2π) while maintaining continuity
func NormalizeAngle(angle float64) float64 {
	// Get the raw modulo
	normalized := math.Mod(angle, 2*math.Pi)
	
	// Handle negative angles by wrapping to [0, 2π)
	if normalized < 0 {
		normalized += 2 * math.Pi
	}
	
	return normalized
}

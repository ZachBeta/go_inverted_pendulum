package env

import (
	"fmt"
	"log"
	"math"
)

// Pendulum represents the inverted pendulum system
type Pendulum struct {
	config Config
	state  State
	logger *log.Logger
	lastForce float64 // Track last applied force
}

// NewPendulum creates a new pendulum system with given config and logger
func NewPendulum(config Config, logger *log.Logger) *Pendulum {
	if logger == nil {
		logger = log.Default()
	}
	
	p := &Pendulum{
		config: config,
		state: State{
			CartPosition: 0,
			CartVelocity: 0,
			AngleRadians: math.Pi, // starting hanging down
			AngularVel:   0,
			TimeStep:     0,
		},
		logger: logger,
	}
	
	p.logger.Printf("Initialized pendulum with config: %+v\n", config)
	return p
}

// GetState returns the current state (immutable)
func (p *Pendulum) GetState() State {
	return p.state
}

// GetConfig returns the current config (immutable)
func (p *Pendulum) GetConfig() Config {
	return p.config
}

// GetLastForce returns the last force applied to the pendulum
func (p *Pendulum) GetLastForce() float64 {
	return p.lastForce
}

// Step advances the simulation by one timestep with the given force
// Returns new state and error if any constraints are violated
func (p *Pendulum) Step(force float64) (State, error) {
	p.lastForce = force // Store force for visualization
	
	// Clamp force to allowed range
	force = math.Max(-p.config.MaxForce, math.Min(force, p.config.MaxForce))
	
	p.logger.Printf("Step %d: Applying force: %.2f\n", p.state.TimeStep, force)

	// Calculate derivatives using equations of motion
	sinTheta := math.Sin(p.state.AngleRadians)
	cosTheta := math.Cos(p.state.AngleRadians)
	
	// Helpful constants
	g := p.config.Gravity
	m := p.config.CartMass
	M := p.config.PendulumMass
	l := p.config.Length
	dt := p.config.DeltaTime
	
	// Calculate accelerations using the full nonlinear equations
	den := m + M*math.Pow(sinTheta, 2)
	
	cartAcc := (force + M*g*sinTheta*cosTheta - M*l*math.Pow(p.state.AngularVel, 2)*sinTheta) / den
	angularAcc := (g*sinTheta*cosTheta - cartAcc*cosTheta) / l

	// Update velocities
	newCartVel := p.state.CartVelocity + cartAcc*dt
	newAngularVel := p.state.AngularVel + angularAcc*dt
	
	// Update positions
	newCartPos := p.state.CartPosition + newCartVel*dt
	newAngle := NormalizeAngle(p.state.AngleRadians + newAngularVel*dt)
	
	// Check track bounds
	if math.Abs(newCartPos) > p.config.TrackLength/2 {
		return p.state, fmt.Errorf("cart position %.2f exceeds track bounds ±%.2f", 
			newCartPos, p.config.TrackLength/2)
	}
	
	// Create new immutable state
	newState := State{
		CartPosition: newCartPos,
		CartVelocity: newCartVel,
		AngleRadians: newAngle,
		AngularVel:   newAngularVel,
		TimeStep:     p.state.TimeStep + 1,
	}
	
	p.logger.Printf("New state: %+v\n", newState)
	
	// Update internal state
	p.state = newState
	
	return newState, nil
}

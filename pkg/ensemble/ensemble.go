// Package ensemble implements a collection of neural networks
// trained in parallel with only the best one being visualized
package ensemble

import (
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/neural"
	"github.com/zachbeta/go_inverted_pendulum/pkg/training"
)

// NetworkInstance represents a single neural network with its trainer and metrics
type NetworkInstance struct {
	ID            int
	Network       *neural.Network
	Trainer       *training.Trainer
	Pendulum      *env.Pendulum
	CurrentTicks  int
	MaxTicks      int
	Episodes      int
	SuccessRate   float64
	AvgReward     float64
	LastHiddenActivation float64
	PrevState     env.State
	Failed        bool
}

// Ensemble manages multiple neural networks trained in parallel
type Ensemble struct {
	Networks       []*NetworkInstance
	BestNetworkIdx int
	Logger         *log.Logger
	Config         Config
	mutex          sync.RWMutex
}

// Config holds ensemble configuration parameters
type Config struct {
	NetworkCount    int
	MutationRate    float64
	CrossoverRate   float64
	SelectionRate   float64
	ReplacementRate float64
}

// NewDefaultConfig returns a default ensemble configuration
func NewDefaultConfig() Config {
	return Config{
		NetworkCount:    10,
		MutationRate:    0.1,
		CrossoverRate:   0.2,
		SelectionRate:   0.3,
		ReplacementRate: 0.1,
	}
}

// NewEnsemble creates a new ensemble of neural networks
func NewEnsemble(config Config, pendulumConfig env.Config, logger *log.Logger) *Ensemble {
	if logger == nil {
		logger = log.Default()
	}

	networks := make([]*NetworkInstance, config.NetworkCount)
	
	for i := 0; i < config.NetworkCount; i++ {
		// Create a new network with slightly different initial weights
		network := neural.NewNetwork()
		network.SetDebug(i == 0) // Only enable debug for the first network
		
		// Add some variation to initial weights
		weights := network.GetWeights()
		for j := range weights {
			// Add small random variations to each network
			variation := (float64(i) / float64(config.NetworkCount) - 0.5) * 0.2
			weights[j] += variation
		}
		network.SetWeights(weights)
		
		// Create trainer with default config
		trainingConfig := training.NewDefaultConfig()
		trainer := training.NewTrainer(trainingConfig, network, logger)
		
		// Create pendulum instance
		pendulum := env.NewPendulum(pendulumConfig, logger)
		
		networks[i] = &NetworkInstance{
			ID:       i,
			Network:  network,
			Trainer:  trainer,
			Pendulum: pendulum,
			PrevState: pendulum.GetState(),
			Failed:   false,
		}
	}
	
	return &Ensemble{
		Networks:       networks,
		BestNetworkIdx: 0,
		Logger:         logger,
		Config:         config,
	}
}

// Step advances all networks by one time step
func (e *Ensemble) Step() error {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	
	// Track if all networks have failed
	allFailed := true
	
	// Process each network
	for i, instance := range e.Networks {
		// Skip networks that have already failed
		if instance.Failed {
			continue
		}
		
		allFailed = false
		
		// Get current state
		state := instance.Pendulum.GetState()
		
		// Get force from network and store hidden activation
		force, hiddenActivation := instance.Network.ForwardWithActivation(state)
		instance.LastHiddenActivation = hiddenActivation
		
		// Apply force and get new state
		newState, err := instance.Pendulum.Step(force)
		
		// Calculate reward for this step
		stepReward := calculateReward(instance.PrevState, state)
		
		// Create experience for training
		experience := training.Experience{
			State:     state,
			Action:    force,
			Reward:    stepReward,
			NextState: newState,
			Done:      err != nil,
			TimeStep:  uint64(instance.CurrentTicks),
		}
		
		// Add experience to trainer
		instance.Trainer.AddExperience(experience)
		
		if err != nil {
			// Mark as failed
			instance.Failed = true
			
			// Handle end of episode
			instance.Trainer.OnEpisodeEnd(instance.CurrentTicks)
			
			// Update max ticks if this was the best episode
			if instance.CurrentTicks > instance.MaxTicks {
				instance.MaxTicks = instance.CurrentTicks
			}
			
			// Reset pendulum for next episode
			config := instance.Pendulum.GetConfig()
			instance.Pendulum = env.NewPendulum(config, e.Logger)
			instance.PrevState = instance.Pendulum.GetState()
			instance.Episodes++
			instance.CurrentTicks = 0
		} else {
			// Update previous state and increment ticks
			instance.PrevState = newState
			instance.CurrentTicks++
		}
		
		// Update stats
		stats := instance.Trainer.GetTrainingStats()
		if sr, ok := stats["success_rate"].(float64); ok {
			instance.SuccessRate = sr
		}
		if ar, ok := stats["avg_reward"].(float64); ok {
			instance.AvgReward = ar
		}
		
		// Update best network index
		if i != e.BestNetworkIdx && instance.MaxTicks > e.Networks[e.BestNetworkIdx].MaxTicks {
			e.Logger.Printf("New best network: #%d with %d ticks (previous best: #%d with %d ticks)",
				i, instance.MaxTicks, e.BestNetworkIdx, e.Networks[e.BestNetworkIdx].MaxTicks)
			e.BestNetworkIdx = i
		}
	}
	
	// If all networks have failed, reset them all
	if allFailed {
		e.evolveNetworks()
	}
	
	return nil
}

// GetBestNetwork returns the best performing network instance
func (e *Ensemble) GetBestNetwork() *NetworkInstance {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	return e.Networks[e.BestNetworkIdx]
}

// GetAllNetworkStats returns statistics for all networks
func (e *Ensemble) GetAllNetworkStats() []map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	
	stats := make([]map[string]interface{}, len(e.Networks))
	
	for i, instance := range e.Networks {
		stats[i] = map[string]interface{}{
			"id":           instance.ID,
			"episodes":     instance.Episodes,
			"current_ticks": instance.CurrentTicks,
			"max_ticks":    instance.MaxTicks,
			"success_rate": instance.SuccessRate,
			"avg_reward":   instance.AvgReward,
			"is_best":      i == e.BestNetworkIdx,
			"status":       getNetworkStatus(instance),
		}
	}
	
	return stats
}

// evolveNetworks implements a simple evolutionary algorithm to improve networks
func (e *Ensemble) evolveNetworks() {
	e.Logger.Printf("Evolving networks after generation completed")
	
	// Sort networks by performance (max ticks)
	sort.Slice(e.Networks, func(i, j int) bool {
		return e.Networks[i].MaxTicks > e.Networks[j].MaxTicks
	})
	
	// Update best network index (should be 0 after sorting)
	e.BestNetworkIdx = 0
	
	// Keep the top networks unchanged
	eliteCount := int(float64(len(e.Networks)) * e.Config.SelectionRate)
	if eliteCount < 1 {
		eliteCount = 1
	}
	
	// Replace worst performers with mutations of the best
	replaceCount := int(float64(len(e.Networks)) * e.Config.ReplacementRate)
	if replaceCount < 1 {
		replaceCount = 1
	}
	
	for i := len(e.Networks) - replaceCount; i < len(e.Networks); i++ {
		// Choose a parent from the elite group
		parentIdx := i % eliteCount
		
		// Copy weights from parent
		parentWeights := e.Networks[parentIdx].Network.GetWeights()
		childWeights := make([]float64, len(parentWeights))
		copy(childWeights, parentWeights)
		
		// Apply mutation
		for j := range childWeights {
			// Add random mutation
			mutation := (rand.Float64() * 2 - 1) * e.Config.MutationRate
			childWeights[j] += mutation
		}
		
		// Update network weights
		e.Networks[i].Network.SetWeights(childWeights)
		
		// Reset pendulum and stats
		config := e.Networks[i].Pendulum.GetConfig()
		e.Networks[i].Pendulum = env.NewPendulum(config, e.Logger)
		e.Networks[i].PrevState = e.Networks[i].Pendulum.GetState()
		e.Networks[i].CurrentTicks = 0
		e.Networks[i].Episodes = 0
		e.Networks[i].Failed = false
	}
	
	// For networks in the middle, apply crossover between elites
	for i := eliteCount; i < len(e.Networks) - replaceCount; i++ {
		// Choose two parents from the elite group
		parent1Idx := i % eliteCount
		parent2Idx := (i + 1) % eliteCount
		
		// Get parent weights
		parent1Weights := e.Networks[parent1Idx].Network.GetWeights()
		parent2Weights := e.Networks[parent2Idx].Network.GetWeights()
		
		// Create child weights through crossover
		childWeights := make([]float64, len(parent1Weights))
		for j := range childWeights {
			// Crossover with random weighting
			alpha := rand.Float64()
			childWeights[j] = alpha*parent1Weights[j] + (1-alpha)*parent2Weights[j]
			
			// Apply small mutation
			mutation := (rand.Float64() * 2 - 1) * e.Config.MutationRate * 0.5
			childWeights[j] += mutation
		}
		
		// Update network weights
		e.Networks[i].Network.SetWeights(childWeights)
		
		// Reset pendulum and stats
		config := e.Networks[i].Pendulum.GetConfig()
		e.Networks[i].Pendulum = env.NewPendulum(config, e.Logger)
		e.Networks[i].PrevState = e.Networks[i].Pendulum.GetState()
		e.Networks[i].CurrentTicks = 0
		e.Networks[i].Episodes = 0
		e.Networks[i].Failed = false
	}
	
	e.Logger.Printf("Network evolution completed. Best network #%d with %d ticks", 
		e.BestNetworkIdx, e.Networks[e.BestNetworkIdx].MaxTicks)
}

// getNetworkStatus returns a string describing the network's status
func getNetworkStatus(instance *NetworkInstance) string {
	if instance.Failed {
		return "failed"
	}
	return "active"
}

// calculateReward computes the reward for a state transition
func calculateReward(prevState, currentState env.State) float64 {
	// Simple reward function based on angle from vertical
	angleReward := 1.0 - math.Abs(currentState.AngleRadians) / math.Pi
	
	// Penalize high angular velocity
	velocityPenalty := math.Min(1.0, math.Abs(currentState.AngularVel) / 10.0) * 0.5
	
	// Penalize cart position far from center
	positionPenalty := math.Min(1.0, math.Abs(currentState.CartPosition) / 2.0) * 0.3
	
	// Combine rewards and penalties
	reward := angleReward - velocityPenalty - positionPenalty
	
	// Scale to [-1, 1] range
	return math.Max(-1.0, math.Min(1.0, reward))
}

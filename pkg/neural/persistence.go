package neural

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// NetworkState represents the serializable state of a neural network
type NetworkState struct {
	Weights      []float64 `json:"weights"`
	LearningRate float64   `json:"learning_rate"`
	SaveTime     string    `json:"save_time"`
	Version      string    `json:"version"`
}

// SaveToFile saves the network state to a JSON file
func (n *Network) SaveToFile(path string) error {
	state := NetworkState{
		Weights:      n.GetWeights(),
		LearningRate: n.learningRate,
		SaveTime:     time.Now().Format(time.RFC3339),
		Version:      "1.0.0",
	}

	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Marshal with indentation for readability
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal network state: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write network state: %w", err)
	}

	if n.debug {
		fmt.Printf("[Network] Saved network state to %s\n", path)
	}
	return nil
}

// LoadFromFile loads the network state from a JSON file
func (n *Network) LoadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read network state: %w", err)
	}

	var state NetworkState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("failed to unmarshal network state: %w", err)
	}

	if err := n.SetWeights(state.Weights); err != nil {
		return fmt.Errorf("failed to set weights: %w", err)
	}
	n.SetLearningRate(state.LearningRate)

	if n.debug {
		fmt.Printf("[Network] Loaded network state from %s (saved at %s)\n", 
			path, state.SaveTime)
	}
	return nil
}

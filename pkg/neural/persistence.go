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
	Weights       []float64 `json:"weights"`
	LearningRate  float64   `json:"learning_rate"`
	SaveTime      string    `json:"save_time"`
	Version       string    `json:"version"`
	MetricsData   *MetricsData `json:"metrics_data,omitempty"`
}

// MetricsData contains information about metrics tracking for this network
type MetricsData struct {
	SessionID     string    `json:"session_id"`
	Episode       int       `json:"episode"`
	Step          int       `json:"step"`
	TrainingStats *TrainingStats `json:"training_stats,omitempty"`
}

// TrainingStats contains summarized training statistics
type TrainingStats struct {
	EpisodeCount  int       `json:"episode_count"`
	SuccessRate   float64   `json:"success_rate"`
	AvgReward     float64   `json:"avg_reward"`
	AvgBalanceTime float64  `json:"avg_balance_time"`
	WeightChanges map[string]float64 `json:"weight_changes,omitempty"`
}

// SaveToFile saves the network state to a JSON file
func (n *Network) SaveToFile(path string) error {
	// Create basic network state
	state := NetworkState{
		Weights:      n.GetWeights(),
		LearningRate: n.learningRate,
		SaveTime:     time.Now().Format(time.RFC3339),
		Version:      "1.0.0",
	}

	// Add metrics data if available
	if n.metrics != nil {
		metricsData := &MetricsData{
			SessionID: n.metrics.GetSessionID(),
			Episode:   n.currentEpisode,
			Step:      n.currentStep,
		}
		
		// Try to get session summary
		if summary, err := n.metrics.GetSessionSummary(); err == nil {
			// Convert summary to TrainingStats
			stats := &TrainingStats{}
			
			if episodeCount, ok := summary["episode_count"].(int); ok {
				stats.EpisodeCount = episodeCount
			}
			
			if successRate, ok := summary["success_rate"].(float64); ok {
				stats.SuccessRate = successRate
			}
			
			if avgReward, ok := summary["avg_reward"].(float64); ok {
				stats.AvgReward = avgReward
			}
			
			if avgBalanceTime, ok := summary["avg_balance_time"].(float64); ok {
				stats.AvgBalanceTime = avgBalanceTime
			}
			
			if weightChanges, ok := summary["weight_changes"].(map[string]interface{}); ok {
				stats.WeightChanges = make(map[string]float64)
				for k, v := range weightChanges {
					if fv, ok := v.(float64); ok {
						stats.WeightChanges[k] = fv
					}
				}
			}
			
			metricsData.TrainingStats = stats
			
			// Log training progress to metrics database
			if n.metrics != nil {
				n.metrics.LogTrainingProgress(
					stats.EpisodeCount, 
					stats.SuccessRate, 
					stats.AvgReward, 
					stats.WeightChanges,
				)
			}
		}
		
		state.MetricsData = metricsData
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

	// Log the save operation to metrics database
	if n.metrics != nil {
		n.metrics.LogNetworkOperation("save", path, true)
	} else if n.debug {
		n.logger.Printf("[Network] Saved network state to %s", path)
	}
	
	return nil
}

// LoadFromFile loads the network state from a JSON file
func (n *Network) LoadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		// Log failed load operation to metrics database
		if n.metrics != nil {
			n.metrics.LogNetworkOperation("load", path, false)
		}
		return fmt.Errorf("failed to read network state: %w", err)
	}

	var state NetworkState
	if err := json.Unmarshal(data, &state); err != nil {
		// Log failed load operation to metrics database
		if n.metrics != nil {
			n.metrics.LogNetworkOperation("load", path, false)
		}
		return fmt.Errorf("failed to unmarshal network state: %w", err)
	}

	// Set weights
	if err := n.SetWeights(state.Weights); err != nil {
		// Log failed load operation to metrics database
		if n.metrics != nil {
			n.metrics.LogNetworkOperation("load", path, false)
		}
		return fmt.Errorf("failed to set weights: %w", err)
	}

	// Set learning rate
	n.SetLearningRate(state.LearningRate)

	// Update metrics data if available
	if state.MetricsData != nil && n.metrics != nil {
		n.currentEpisode = state.MetricsData.Episode
		n.currentStep = state.MetricsData.Step
		n.metrics.SetEpisode(state.MetricsData.Episode)
		
		// If training stats are available, log them to metrics database
		if state.MetricsData.TrainingStats != nil {
			stats := state.MetricsData.TrainingStats
			n.metrics.LogTrainingProgress(
				stats.EpisodeCount,
				stats.SuccessRate,
				stats.AvgReward,
				stats.WeightChanges,
			)
		}
	}

	// Log successful load operation to metrics database
	if n.metrics != nil {
		n.metrics.LogNetworkOperation("load", path, true)
	} else if n.debug {
		n.logger.Printf("[Network] Loaded network state from %s (saved at %s)", 
			path, state.SaveTime)
	}
	
	return nil
}

// SaveCheckpoint saves the current network state to a checkpoint file
func (n *Network) SaveCheckpoint(checkpointDir string) error {
	// Create checkpoint filename based on episode and timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("checkpoint_ep%d_%s.json", n.currentEpisode, timestamp)
	path := filepath.Join(checkpointDir, filename)
	
	err := n.SaveToFile(path)
	if err != nil {
		// Log failed checkpoint operation to metrics database
		if n.metrics != nil {
			n.metrics.LogNetworkOperation("checkpoint", path, false)
		}
		return fmt.Errorf("failed to save checkpoint: %w", err)
	}
	
	// Log successful checkpoint operation to metrics database
	if n.metrics != nil {
		n.metrics.LogNetworkOperation("checkpoint", path, true)
	} else if n.debug {
		n.logger.Printf("[Network] Saved checkpoint to %s", path)
	}
	
	return nil
}

// LoadLatestCheckpoint loads the most recent checkpoint from a directory
func (n *Network) LoadLatestCheckpoint(checkpointDir string) error {
	// Ensure directory exists
	if _, err := os.Stat(checkpointDir); os.IsNotExist(err) {
		return fmt.Errorf("checkpoint directory does not exist: %s", checkpointDir)
	}
	
	// Get all checkpoint files
	files, err := os.ReadDir(checkpointDir)
	if err != nil {
		return fmt.Errorf("failed to read checkpoint directory: %w", err)
	}
	
	// Find the latest checkpoint file
	var latestFile string
	var latestTime time.Time
	
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		
		// Check if file is a checkpoint file
		if filepath.Ext(file.Name()) != ".json" || len(file.Name()) < 10 {
			continue
		}
		
		fileInfo, err := file.Info()
		if err != nil {
			continue
		}
		
		modTime := fileInfo.ModTime()
		if latestFile == "" || modTime.After(latestTime) {
			latestFile = file.Name()
			latestTime = modTime
		}
	}
	
	if latestFile == "" {
		return fmt.Errorf("no checkpoint files found in directory: %s", checkpointDir)
	}
	
	// Load the latest checkpoint
	path := filepath.Join(checkpointDir, latestFile)
	err = n.LoadFromFile(path)
	if err != nil {
		return fmt.Errorf("failed to load latest checkpoint: %w", err)
	}
	
	return nil
}

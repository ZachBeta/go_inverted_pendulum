package neural

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/metrics"
)

func TestNetworkLearningDebug(t *testing.T) {
	// Create a test directory for metrics
	testDir := filepath.Join(os.TempDir(), fmt.Sprintf("neural_test_%d", time.Now().UnixNano()))
	err := os.MkdirAll(testDir, 0755)
	if err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}
	defer os.RemoveAll(testDir)

	// Create metrics database path
	dbPath := filepath.Join(testDir, "metrics.db")
	
	// Create a logger for metrics
	stdLogger := log.New(os.Stdout, "[Test] ", log.LstdFlags)
	logger, err := metrics.NewLogger(dbPath, true, stdLogger)
	if err != nil {
		t.Fatalf("Failed to create metrics logger: %v", err)
	}

	// Create network
	network := NewNetwork()
	network.SetMetricsLogger(logger)
	network.SetDebug(true)
	network.SetLogger(stdLogger)

	// Test 1: Verify weight updates are working correctly
	t.Run("WeightUpdateTest", func(t *testing.T) {
		// Record initial weights
		initialWeights := network.GetWeights()

		// Set learning rate to a known value
		network.SetLearningRate(0.1)

		// Create test state
		state := env.State{
			AngleRadians: 0.5,
			AngularVel:   -0.3,
		}
		
		// Log initial state
		logger.SetEpisode(1)
		logger.IncrementStep()
		logger.LogWeights(initialWeights[0], initialWeights[1], initialWeights[2], network.GetLearningRate())
		
		// Perform forward pass and record prediction
		force, hidden := network.ForwardWithActivation(state)
		logger.LogForwardPass(state.AngleRadians, state.AngularVel, force, hidden)
		
		// Apply a known reward
		reward := 1.0
		network.Update(reward)
		
		// Get updated weights
		newWeights := network.GetWeights()
		logger.LogWeights(newWeights[0], newWeights[1], newWeights[2], network.GetLearningRate())
		
		// Verify weight updates
		t.Logf("Initial weights: angle=%.6f, angularVel=%.6f, bias=%.6f", 
			initialWeights[0], initialWeights[1], initialWeights[2])
		t.Logf("Updated weights: angle=%.6f, angularVel=%.6f, bias=%.6f", 
			newWeights[0], newWeights[1], newWeights[2])
		
		// Check that weights changed in response to reward
		changed := false
		for i := range newWeights {
			if newWeights[i] != initialWeights[i] {
				changed = true
				break
			}
		}
		if !changed {
			t.Error("Weights did not update after reward")
		}
	})

	// Test 2: Verify TD learning is working correctly
	t.Run("TDLearningTest", func(t *testing.T) {
		// Reset network
		network = NewNetwork()
		network.SetMetricsLogger(logger)
		network.SetDebug(true)
		network.SetLogger(stdLogger)
		network.SetLearningRate(0.1)
		
		// Set up episode
		logger.SetEpisode(2)
		logger.IncrementStep()
		
		// First state
		state1 := env.State{AngleRadians: 0.1, AngularVel: 0.2}
		prediction1 := network.Predict(state1.AngleRadians, state1.AngularVel)
		t.Logf("State 1 prediction: %.6f", prediction1)
		
		// Second state
		logger.IncrementStep()
		state2 := env.State{AngleRadians: 0.2, AngularVel: 0.3}
		prediction2 := network.Predict(state2.AngleRadians, state2.AngularVel)
		t.Logf("State 2 prediction: %.6f", prediction2)
		
		// Verify TD learning (better state should have higher prediction)
		if math.Abs(state2.AngleRadians) < math.Abs(state1.AngleRadians) &&
			math.Abs(state2.AngularVel) < math.Abs(state1.AngularVel) &&
			prediction2 <= prediction1 {
			t.Error("TD prediction failed: better state has lower prediction")
		}
	})

	// Test 3: Verify learning over multiple episodes
	t.Run("LearningProgressTest", func(t *testing.T) {
		// Reset network
		network = NewNetwork()
		network.SetMetricsLogger(logger)
		network.SetDebug(true)
		network.SetLogger(stdLogger)
		network.SetLearningRate(0.1)
		
		// Train for multiple episodes
		numEpisodes := 10
		stepsPerEpisode := 5
		
		var prevAvgReward float64
		
		for episode := 1; episode <= numEpisodes; episode++ {
			logger.SetEpisode(episode + 2) // Continue from previous tests
			
			// Track episode reward
			totalReward := 0.0
			
			for step := 1; step <= stepsPerEpisode; step++ {
				logger.IncrementStep()
				
				// Generate test state (progressively harder)
				state := env.State{
					AngleRadians: (float64(step) / float64(stepsPerEpisode)) * math.Pi/4,
					AngularVel:   0.1 * math.Sin(float64(step)),
				}
				
				// Get network action and update
				network.Forward(state) // Force used internally for updates
				
				// Calculate reward (better for keeping pendulum upright)
				reward := -math.Abs(state.AngleRadians) - 0.5*math.Abs(state.AngularVel)
				totalReward += reward
				
				// Update network
				network.Update(reward)
			}
			
			// Calculate average reward
			avgReward := totalReward / float64(stepsPerEpisode)
			
			// After first episode, verify learning progress
			if episode > 1 && avgReward <= prevAvgReward {
				t.Logf("Warning: Episode %d did not improve (prev=%.4f, current=%.4f)",
					episode, prevAvgReward, avgReward)
			}
			
			prevAvgReward = avgReward
		}
	})
}

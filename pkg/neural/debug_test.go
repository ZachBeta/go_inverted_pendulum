package neural

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/zachbeta/go_inverted_pendulum/pkg/metrics"
)

// TestNetworkLearningDebug is a comprehensive test to debug why the network isn't learning
// It creates a controlled environment to test various aspects of the learning process
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
	network.debug = true
	network.logger = stdLogger

	// Test 1: Verify weight updates are working correctly
	t.Run("WeightUpdateTest", func(t *testing.T) {
		// Record initial weights
		initialAngleWeight := network.angleWeight
		initialAngularVelWeight := network.angularVelWeight
		initialBias := network.bias

		// Set learning rate to a known value
		network.SetLearningRate(0.1)

		// Perform a controlled forward pass with known inputs
		angle := 0.5
		angularVel := -0.3
		inputs := []float64{angle, angularVel}
		
		// Log initial state
		logger.SetEpisode(1)
		logger.SetStep(1)
		logger.LogWeights(network.angleWeight, network.angularVelWeight, network.bias, network.learningRate)
		
		// Perform forward pass and record prediction
		prediction := network.Forward(inputs)
		logger.LogForwardPass(inputs, prediction)
		
		// Apply a known reward
		reward := 1.0
		network.Update(reward)
		
		// Log updated weights
		logger.LogWeights(network.angleWeight, network.angularVelWeight, network.bias, network.learningRate)
		
		// Verify weight updates
		expectedAngleUpdate := network.learningRate * reward * (-angle) * sign(prediction)
		expectedAngularVelUpdate := network.learningRate * reward * (-angularVel) * sign(prediction)
		expectedBiasUpdate := network.learningRate * reward * sign(prediction)
		
		t.Logf("Initial weights: angle=%.6f, angularVel=%.6f, bias=%.6f", 
			initialAngleWeight, initialAngularVelWeight, initialBias)
		t.Logf("Updated weights: angle=%.6f, angularVel=%.6f, bias=%.6f", 
			network.angleWeight, network.angularVelWeight, network.bias)
		t.Logf("Weight changes: angle=%.6f, angularVel=%.6f, bias=%.6f", 
			network.angleWeight-initialAngleWeight, 
			network.angularVelWeight-initialAngularVelWeight, 
			network.bias-initialBias)
		t.Logf("Expected changes: angle=%.6f, angularVel=%.6f, bias=%.6f", 
			expectedAngleUpdate, expectedAngularVelUpdate, expectedBiasUpdate)
		
		// Check if weight updates match expectations (within a small epsilon)
		epsilon := 1e-6
		if math.Abs((network.angleWeight-initialAngleWeight)-expectedAngleUpdate) > epsilon {
			t.Errorf("Angle weight update incorrect: got %.6f, expected %.6f", 
				network.angleWeight-initialAngleWeight, expectedAngleUpdate)
		}
		if math.Abs((network.angularVelWeight-initialAngularVelWeight)-expectedAngularVelUpdate) > epsilon {
			t.Errorf("Angular velocity weight update incorrect: got %.6f, expected %.6f", 
				network.angularVelWeight-initialAngularVelWeight, expectedAngularVelUpdate)
		}
		if math.Abs((network.bias-initialBias)-expectedBiasUpdate) > epsilon {
			t.Errorf("Bias update incorrect: got %.6f, expected %.6f", 
				network.bias-initialBias, expectedBiasUpdate)
		}
	})

	// Test 2: Verify TD learning is working correctly
	t.Run("TDLearningTest", func(t *testing.T) {
		// Reset network
		network = NewNetwork()
		network.SetMetricsLogger(logger)
		network.debug = true
		network.logger = stdLogger
		network.SetLearningRate(0.1)
		
		// Set up episode
		logger.SetEpisode(2)
		logger.SetStep(1)
		
		// First state
		state1 := []float64{0.1, 0.2}
		prediction1 := network.Predict(state1)
		t.Logf("State 1 prediction: %.6f", prediction1)
		
		// Second state with reward
		logger.SetStep(2)
		state2 := []float64{0.2, 0.3}
		prediction2 := network.Predict(state2)
		t.Logf("State 2 prediction: %.6f", prediction2)
		
		// Apply TD update (reward + gamma * prediction2 - prediction1)
		reward := 0.5
		gamma := 0.9
		tdTarget := reward + gamma*prediction2
		tdError := tdTarget - prediction1
		
		t.Logf("TD target: %.6f", tdTarget)
		t.Logf("TD error: %.6f", tdError)
		
		// Update network with the reward
		network.Update(reward)
		
		// Verify TD error is logged correctly
		// This is more of a sanity check since we can't directly access the TD error from the network
		analysis, err := logger.AnalyzePredictionAccuracy(2)
		if err != nil {
			t.Fatalf("Failed to analyze prediction accuracy: %v", err)
		}
		
		t.Logf("Prediction analysis: %+v", analysis)
	})

	// Test 3: Verify learning over multiple episodes
	t.Run("LearningProgressTest", func(t *testing.T) {
		// Reset network
		network = NewNetwork()
		network.SetMetricsLogger(logger)
		network.debug = true
		network.logger = stdLogger
		network.SetLearningRate(0.1)
		
		// Train for multiple episodes
		numEpisodes := 10
		stepsPerEpisode := 5
		
		// Define a simple environment for testing
		// The goal is to learn that positive angles should be pushed right (positive force)
		// and negative angles should be pushed left (negative force)
		for episode := 1; episode <= numEpisodes; episode++ {
			logger.SetEpisode(episode + 2) // Continue from previous tests
			
			// Track episode reward
			totalReward := 0.0
			
			for step := 1; step <= stepsPerEpisode; step++ {
				logger.SetStep(step)
				
				// Generate a test state
				angle := (float64(step) / float64(stepsPerEpisode)) * math.Pi - (math.Pi / 2)
				angularVel := 0.1 * math.Sin(float64(step))
				state := []float64{angle, angularVel}
				
				// Get network prediction and action
				prediction := network.Predict(state)
				action := network.Forward(state)
				
				// Determine correct action and reward
				correctAction := 1.0
				if angle < 0 {
					correctAction = -1.0
				}
				
				// Calculate reward based on how close the action is to correct action
				reward := 0.0
				if math.Signbit(action) == math.Signbit(correctAction) {
					reward = 1.0 - 0.5*math.Abs(action-correctAction)
				} else {
					reward = -0.5
				}
				
				totalReward += reward
				
				// Update network
				network.Update(reward)
				
				// Log detailed information
				t.Logf("Episode %d, Step %d: angle=%.2f, prediction=%.4f, action=%.4f, reward=%.4f", 
					episode+2, step, angle, prediction, action, reward)
			}
			
			// Log episode results
			avgReward := totalReward / float64(stepsPerEpisode)
			logger.LogEpisodeResult(avgReward, avgReward > 0.5)
			t.Logf("Episode %d complete, average reward: %.4f", episode+2, avgReward)
			
			// Test final weights
			if episode == numEpisodes {
				t.Logf("Final weights: angle=%.6f, angularVel=%.6f, bias=%.6f", 
					network.angleWeight, network.angularVelWeight, network.bias)
			}
		}
		
		// Analyze learning progress
		progress, err := logger.AnalyzeLearningProgress(numEpisodes)
		if err != nil {
			t.Fatalf("Failed to analyze learning progress: %v", err)
		}
		
		t.Logf("Learning progress analysis: %+v", progress)
		
		// Check for learning issues
		issues, err := logger.DetectLearningIssues()
		if err != nil {
			t.Fatalf("Failed to detect learning issues: %v", err)
		}
		
		t.Logf("Learning issues: %+v", issues)
		
		// Verify the network has learned the correct policy
		// Test with positive angle
		positiveState := []float64{0.5, 0.0}
		positiveAction := network.Forward(positiveState)
		if positiveAction <= 0 {
			t.Errorf("Network failed to learn correct policy for positive angle: got %.4f, expected > 0", 
				positiveAction)
		}
		
		// Test with negative angle
		negativeState := []float64{-0.5, 0.0}
		negativeAction := network.Forward(negativeState)
		if negativeAction >= 0 {
			t.Errorf("Network failed to learn correct policy for negative angle: got %.4f, expected < 0", 
				negativeAction)
		}
	})

	// Output the path to the metrics database for further analysis
	t.Logf("Metrics database created at: %s", dbPath)
	t.Logf("Use the debug tool to analyze: go run cmd/debug/main.go -db %s", dbPath)
}

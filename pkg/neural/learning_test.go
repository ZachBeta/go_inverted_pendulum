package neural

import (
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/metrics"
)

// TestNetworkLearnsToBalance tests that the network progressively learns to balance the pendulum
func TestNetworkLearnsToBalance(t *testing.T) {
	// Given a new network with default weights
	network := NewNetwork()
	network.SetDebug(false) // Disable debug output during tests
	
	// And a sequence of training episodes with progressively better states
	episodes := []struct {
		description string
		state       env.State
		reward      float64
		nextState   env.State
	}{
		{
			description: "Starting with large angle",
			state: env.State{AngleRadians: 0.5, AngularVel: 0.8},
			reward: -0.7,
			nextState: env.State{AngleRadians: 0.4, AngularVel: 0.6},
		},
		{
			description: "Improving to medium angle",
			state: env.State{AngleRadians: 0.3, AngularVel: 0.4},
			reward: -0.3,
			nextState: env.State{AngleRadians: 0.2, AngularVel: 0.2},
		},
		{
			description: "Nearly balanced",
			state: env.State{AngleRadians: 0.1, AngularVel: 0.1},
			reward: 0.8,
			nextState: env.State{AngleRadians: 0.05, AngularVel: 0.05},
		},
	}
	
	// When we train the network through these episodes
	initialPrediction := network.Predict(episodes[0].state.AngleRadians, episodes[0].state.AngularVel)
	
	for _, episode := range episodes {
		// Forward pass to get action
		_ = network.Forward(episode.state)
		
		// Update with reward
		network.Update(episode.reward)
		
		// TD learning update
		currentValue := network.Predict(episode.state.AngleRadians, episode.state.AngularVel)
		nextValue := network.Predict(episode.nextState.AngleRadians, episode.nextState.AngularVel)
		
		// Verify TD learning improves value estimates
		if episode.reward > 0 && nextValue <= currentValue {
			t.Errorf("%s: TD learning failed to increase value for positive reward", episode.description)
		}
		if episode.reward < 0 && nextValue >= currentValue {
			t.Errorf("%s: TD learning failed to decrease value for negative reward", episode.description)
		}
	}
	
	// Then the network should have improved its prediction for the initial state
	finalPrediction := network.Predict(episodes[0].state.AngleRadians, episodes[0].state.AngularVel)
	
	// The network should now predict a better value for the initial state
	if finalPrediction <= initialPrediction {
		t.Errorf("Network failed to learn: initial prediction %.4f, final prediction %.4f", 
			initialPrediction, finalPrediction)
	}
}

// TestTemporalDifferencePredictions verifies that the network's TD predictions
// accurately reflect state quality
func TestTemporalDifferencePredictions(t *testing.T) {
	// Given a trained network
	network := NewNetwork()
	network.SetDebug(false) // Disable debug output during tests
	
	// Train network with basic scenarios (simplified for test)
	scenarios := []struct {
		state  env.State
		reward float64
	}{
		// Good states (balanced)
		{env.State{AngleRadians: 0.01, AngularVel: 0.01}, 0.9},
		{env.State{AngleRadians: -0.01, AngularVel: -0.01}, 0.9},
		
		// Bad states (falling)
		{env.State{AngleRadians: 0.5, AngularVel: 1.0}, -0.8},
		{env.State{AngleRadians: -0.5, AngularVel: -1.0}, -0.8},
	}
	
	// Train network on scenarios
	for _, s := range scenarios {
		network.Forward(s.state)
		network.Update(s.reward)
	}
	
	// When we evaluate state values
	tests := []struct {
		description string
		state       env.State
		betterState env.State
		worseState  env.State
	}{
		{
			description: "Balanced state should be valued higher than tilted state",
			state:       env.State{AngleRadians: 0.1, AngularVel: 0.1},
			betterState: env.State{AngleRadians: 0.05, AngularVel: 0.05},
			worseState:  env.State{AngleRadians: 0.2, AngularVel: 0.2},
		},
		{
			description: "Slower velocity should be valued higher when angle is same",
			state:       env.State{AngleRadians: 0.2, AngularVel: 0.3},
			betterState: env.State{AngleRadians: 0.2, AngularVel: 0.1},
			worseState:  env.State{AngleRadians: 0.2, AngularVel: 0.5},
		},
	}
	
	// Then predictions should reflect state quality
	for _, tt := range tests {
		stateValue := network.Predict(tt.state.AngleRadians, tt.state.AngularVel)
		betterValue := network.Predict(tt.betterState.AngleRadians, tt.betterState.AngularVel)
		worseValue := network.Predict(tt.worseState.AngleRadians, tt.worseState.AngularVel)
		
		if betterValue <= stateValue {
			t.Errorf("%s: better state not valued higher (better=%.4f, current=%.4f)",
				tt.description, betterValue, stateValue)
		}
		
		if worseValue >= stateValue {
			t.Errorf("%s: worse state not valued lower (worse=%.4f, current=%.4f)",
				tt.description, worseValue, stateValue)
		}
	}
}

// TestNetworkImprovesThroughCheckpoints verifies that network performance improves
// across saved and restored checkpoints
func TestNetworkImprovesThroughCheckpoints(t *testing.T) {
	// Create temp directory for checkpoints
	tmpDir, err := os.MkdirTemp("", "checkpoint_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	
	// Given a new network
	network := NewNetwork()
	network.SetDebug(false) // Disable debug output during tests
	
	// Define a standard evaluation function
	evaluateNetwork := func(net *Network) float64 {
		// Run 10 episodes and return average reward
		totalReward := 0.0
		const episodes = 10
		const timeSteps = 100
		
		for i := 0; i < episodes; i++ {
			// Create a pendulum simulation
			pendulum := env.NewPendulum(env.NewDefaultConfig(), nil)
			episodeReward := 0.0
			
			for j := 0; j < timeSteps; j++ {
				state := pendulum.GetState()
				force := net.Forward(state)
				newState, err := pendulum.Step(force)
				if err != nil {
					// Skip this step if we hit a constraint
					continue
				}
				
				// Calculate reward (simplified for test)
				// Higher reward for more upright pendulum
				reward := 1.0 - math.Abs(newState.AngleRadians) / math.Pi
				episodeReward += reward
			}
			
			totalReward += episodeReward / float64(timeSteps)
		}
		
		return totalReward / float64(episodes)
	}
	
	// Measure initial performance
	initialPerformance := evaluateNetwork(network)
	
	// Train for first phase
	for i := 0; i < 100; i++ {
		// Generate experience and train
		pendulum := env.NewPendulum(env.NewDefaultConfig(), nil)
		state := pendulum.GetState()
		force := network.Forward(state)
		newState, err := pendulum.Step(force)
		if err == nil {
			// Calculate reward
			reward := 1.0 - math.Abs(newState.AngleRadians) / math.Pi
			network.Update(reward)
		}
	}
	
	// Save network state
	checkpoint1Path := filepath.Join(tmpDir, "checkpoint1.json")
	if err := network.SaveToFile(checkpoint1Path); err != nil {
		t.Fatalf("Failed to save checkpoint: %v", err)
	}
	
	// Measure performance after first phase
	phase1Performance := evaluateNetwork(network)
	
	// Create new network for second phase
	network2 := NewNetwork()
	network2.SetDebug(false)
	
	// Load checkpoint
	if err := network2.LoadFromFile(checkpoint1Path); err != nil {
		t.Fatalf("Failed to load checkpoint: %v", err)
	}
	
	// Train for second phase
	for i := 0; i < 100; i++ {
		// Generate experience and train
		pendulum := env.NewPendulum(env.NewDefaultConfig(), nil)
		state := pendulum.GetState()
		force := network2.Forward(state)
		newState, err := pendulum.Step(force)
		if err == nil {
			// Calculate reward
			reward := 1.0 - math.Abs(newState.AngleRadians) / math.Pi
			network2.Update(reward)
		}
	}
	
	// Save second checkpoint
	checkpoint2Path := filepath.Join(tmpDir, "checkpoint2.json")
	if err := network2.SaveToFile(checkpoint2Path); err != nil {
		t.Fatalf("Failed to save checkpoint: %v", err)
	}
	
	// Measure final performance
	phase2Performance := evaluateNetwork(network2)
	
	// Then performance should improve across phases
	if phase1Performance <= initialPerformance {
		t.Errorf("Network did not improve in phase 1: initial=%.4f, phase1=%.4f",
			initialPerformance, phase1Performance)
	}
	
	if phase2Performance <= phase1Performance {
		t.Errorf("Network did not improve in phase 2: phase1=%.4f, phase2=%.4f",
			phase1Performance, phase2Performance)
	}
	
	// Verify checkpoint contains expected data
	checkpointData, err := os.ReadFile(checkpoint2Path)
	if err != nil {
		t.Fatalf("Failed to read checkpoint: %v", err)
	}
	
	var checkpoint map[string]interface{}
	if err := json.Unmarshal(checkpointData, &checkpoint); err != nil {
		t.Fatalf("Failed to parse checkpoint JSON: %v", err)
	}
	
	// Check for required fields
	requiredFields := []string{"weights", "learning_rate", "save_time", "version"}
	for _, field := range requiredFields {
		if _, ok := checkpoint[field]; !ok {
			t.Errorf("Checkpoint missing required field: %s", field)
		}
	}
}

// TestLearningMetricsImproveOverTime verifies that key learning metrics improve with training
func TestLearningMetricsImproveOverTime(t *testing.T) {
	// Create temp directory for metrics database
	tmpDir, err := os.MkdirTemp("", "metrics_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	
	// Create metrics logger
	metricsDB := filepath.Join(tmpDir, "test_metrics.db")
	metricsLogger, err := metrics.NewLogger(metricsDB, false, log.Default())
	if err != nil {
		t.Fatalf("Failed to create metrics logger: %v", err)
	}
	defer metricsLogger.Close()
	
	// Given a network
	network := NewNetwork()
	network.SetDebug(false) // Disable debug output during tests
	network.SetMetricsLogger(metricsLogger)
	
	// When we train for multiple episodes
	const episodeCount = 50
	
	// Run training episodes
	for i := 0; i < episodeCount; i++ {
		// Set current episode in network
		network.SetEpisode(i)
		
		// Simulate an episode with pendulum environment
		pendulum := env.NewPendulum(env.NewDefaultConfig(), nil)
		
		// Run for fixed time steps
		const timeSteps = 200
		totalReward := 0.0
		balanceTime := 0
		maxAngle := 0.0
		
		for j := 0; j < timeSteps; j++ {
			// Increment step counter
			network.IncrementStep()
			
			state := pendulum.GetState()
			
			// Track max angle
			if math.Abs(state.AngleRadians) > maxAngle {
				maxAngle = math.Abs(state.AngleRadians)
			}
			
			// Get action from network
			force := network.Forward(state)
			
			// Apply action and get reward
			newState, err := pendulum.Step(force)
			if err != nil {
				continue
			}
			
			// Calculate reward (simplified for test)
			reward := 1.0 - math.Abs(newState.AngleRadians) / math.Pi
			totalReward += reward
			
			// Count time balanced (angle < 0.1 radians)
			if math.Abs(newState.AngleRadians) < 0.1 {
				balanceTime++
			}
			
			// Update network
			network.Update(reward)
		}
		
		// Record episode results in metrics
		success := balanceTime > timeSteps/2 // Success if balanced more than half the time
		metricsLogger.LogEpisodeResult(totalReward, balanceTime, maxAngle, timeSteps, success)
		
		// Every 10 episodes, verify improvement
		if i > 10 && i % 10 == 0 {
			// Get session summary from metrics
			_, err := metricsLogger.GetSessionSummary()
			if err != nil {
				t.Fatalf("Failed to get session summary: %v", err)
			}
			
			// Get early and recent episodes
			earlyEpisodes := make([]map[string]interface{}, 0, 10)
			recentEpisodes := make([]map[string]interface{}, 0, 10)
			
			for ep := 0; ep < 10; ep++ {
				epData, err := metricsLogger.GetEpisodeData(ep)
				if err != nil {
					t.Fatalf("Failed to get episode data: %v", err)
				}
				earlyEpisodes = append(earlyEpisodes, epData)
			}
			
			for ep := i-9; ep <= i; ep++ {
				epData, err := metricsLogger.GetEpisodeData(ep)
				if err != nil {
					t.Fatalf("Failed to get episode data: %v", err)
				}
				recentEpisodes = append(recentEpisodes, epData)
			}
			
			// Calculate metrics from episode data
			earlyAvgReward := calculateAverage(earlyEpisodes, "total_reward")
			recentAvgReward := calculateAverage(recentEpisodes, "total_reward")
			
			earlyAvgBalanceTime := calculateAverage(earlyEpisodes, "balance_time")
			recentAvgBalanceTime := calculateAverage(recentEpisodes, "balance_time")
			
			earlyAvgMaxAngle := calculateAverage(earlyEpisodes, "max_angle")
			recentAvgMaxAngle := calculateAverage(recentEpisodes, "max_angle")
			
			// Then metrics should show improvement
			if recentAvgReward <= earlyAvgReward {
				t.Logf("Warning: Average reward did not improve: early=%.4f, recent=%.4f", 
					earlyAvgReward, recentAvgReward)
			}
			
			if recentAvgBalanceTime <= earlyAvgBalanceTime {
				t.Logf("Warning: Average balance time did not improve: early=%.4f, recent=%.4f", 
					earlyAvgBalanceTime, recentAvgBalanceTime)
			}
			
			if recentAvgMaxAngle >= earlyAvgMaxAngle {
				t.Logf("Warning: Average max angle did not decrease: early=%.4f, recent=%.4f", 
					earlyAvgMaxAngle, recentAvgMaxAngle)
			}
			
			// At least one metric should improve
			if recentAvgReward <= earlyAvgReward && 
			   recentAvgBalanceTime <= earlyAvgBalanceTime && 
			   recentAvgMaxAngle >= earlyAvgMaxAngle {
				t.Errorf("No metrics improved after %d episodes", i)
			}
		}
	}
	
	// Get final session summary
	summary, err := metricsLogger.GetSessionSummary()
	if err != nil {
		t.Fatalf("Failed to get final session summary: %v", err)
	}
	
	// Get early and final episodes
	earlyEpisodes := make([]map[string]interface{}, 0, 10)
	finalEpisodes := make([]map[string]interface{}, 0, 10)
	
	for ep := 0; ep < 10; ep++ {
		epData, err := metricsLogger.GetEpisodeData(ep)
		if err != nil {
			t.Fatalf("Failed to get episode data: %v", err)
		}
		earlyEpisodes = append(earlyEpisodes, epData)
	}
	
	for ep := episodeCount-10; ep < episodeCount; ep++ {
		epData, err := metricsLogger.GetEpisodeData(ep)
		if err != nil {
			t.Fatalf("Failed to get episode data: %v", err)
		}
		finalEpisodes = append(finalEpisodes, epData)
	}
	
	// Calculate metrics from episode data
	earlyAvgReward := calculateAverage(earlyEpisodes, "total_reward")
	finalAvgReward := calculateAverage(finalEpisodes, "total_reward")
	
	earlyAvgBalanceTime := calculateAverage(earlyEpisodes, "balance_time")
	finalAvgBalanceTime := calculateAverage(finalEpisodes, "balance_time")
	
	earlyAvgMaxAngle := calculateAverage(earlyEpisodes, "max_angle")
	finalAvgMaxAngle := calculateAverage(finalEpisodes, "max_angle")
	
	// Log final improvement
	t.Logf("Reward improvement: %.4f → %.4f (%.2f%%)", 
		earlyAvgReward, finalAvgReward, 
		(finalAvgReward-earlyAvgReward)/math.Abs(earlyAvgReward)*100)
	
	t.Logf("Balance time improvement: %.4f → %.4f (%.2f%%)", 
		earlyAvgBalanceTime, finalAvgBalanceTime, 
		(finalAvgBalanceTime-earlyAvgBalanceTime)/math.Max(0.001, earlyAvgBalanceTime)*100)
	
	t.Logf("Max angle improvement: %.4f → %.4f (%.2f%%)", 
		earlyAvgMaxAngle, finalAvgMaxAngle, 
		(earlyAvgMaxAngle-finalAvgMaxAngle)/earlyAvgMaxAngle*100)
	
	// Verify metrics were properly stored
	t.Logf("Total episodes recorded: %d", summary["episode_count"])
	if successRate, ok := summary["success_rate"].(float64); ok {
		t.Logf("Success rate: %.2f%%", successRate*100)
	}
}

// calculateAverage calculates the average value of a specific field across episode data
func calculateAverage(episodes []map[string]interface{}, field string) float64 {
	if len(episodes) == 0 {
		return 0
	}
	
	sum := 0.0
	count := 0
	
	for _, ep := range episodes {
		if val, ok := ep[field].(float64); ok {
			sum += val
			count++
		} else if valInt, ok := ep[field].(int); ok {
			sum += float64(valInt)
			count++
		}
	}
	
	if count == 0 {
		return 0
	}
	
	return sum / float64(count)
}

// TestProgressiveLearningBehavior tests the progressive learning behavior of the network
func TestProgressiveLearningBehavior(t *testing.T) {
	network := NewNetwork()
	
	// Test sequence of increasingly challenging states
	states := []struct {
		state  env.State
		reward float64
		desc   string
	}{
		{
			state:  env.State{AngleRadians: 0.1, AngularVel: 0.2},
			reward: 0.8,
			desc:   "Small angle correction",
		},
		{
			state:  env.State{AngleRadians: 0.3, AngularVel: 0.4},
			reward: 0.5,
			desc:   "Medium angle correction",
		},
		{
			state:  env.State{AngleRadians: 0.5, AngularVel: 0.6},
			reward: 0.2,
			desc:   "Large angle correction",
		},
	}
	
	// Track learning progress
	var prevValue float64
	for i, tc := range states {
		// Get network's prediction before update
		initialValue := network.Predict(tc.state.AngleRadians, tc.state.AngularVel)
		
		// Perform forward pass and update
		network.Forward(tc.state) // Force used internally for weight updates
		network.Update(tc.reward)
		
		// Get prediction after update
		newValue := network.Predict(tc.state.AngleRadians, tc.state.AngularVel)
		
		// For non-first states, check learning progression
		if i > 0 {
			valueDiff := newValue - prevValue
			if valueDiff <= 0 {
				t.Errorf("%s: Learning did not progress (value diff: %.4f)", tc.desc, valueDiff)
			}
		}
		prevValue = newValue
		
		// Verify value improved after update
		if newValue <= initialValue {
			t.Errorf("%s: Value did not improve after update (before: %.4f, after: %.4f)", 
				tc.desc, initialValue, newValue)
		}
	}
}

// TestTemporalDifferencePrediction tests the temporal difference prediction of the network
func TestTemporalDifferencePrediction(t *testing.T) {
	network := NewNetwork()
	
	// Test TD prediction accuracy
	scenarios := []struct {
		current     env.State
		next        env.State
		reward      float64
		wantBetter  bool
		desc        string
	}{
		{
			current:    env.State{AngleRadians: 0.3, AngularVel: 0.4},
			next:       env.State{AngleRadians: 0.2, AngularVel: 0.3},
			reward:     0.7,
			wantBetter: true,
			desc:       "Improving state",
		},
		{
			current:    env.State{AngleRadians: 0.1, AngularVel: 0.2},
			next:       env.State{AngleRadians: 0.2, AngularVel: 0.3},
			reward:     -0.3,
			wantBetter: false,
			desc:       "Worsening state",
		},
	}
	
	for _, tc := range scenarios {
		// Get prediction for current state
		currentValue := network.Predict(tc.current.AngleRadians, tc.current.AngularVel)
		
		// Get prediction for next state
		nextValue := network.Predict(tc.next.AngleRadians, tc.next.AngularVel)
		
		if tc.wantBetter && nextValue <= currentValue {
			t.Errorf("%s: Next state value (%.4f) not better than current (%.4f)", 
				tc.desc, nextValue, currentValue)
		} else if !tc.wantBetter && nextValue >= currentValue {
			t.Errorf("%s: Next state value (%.4f) not worse than current (%.4f)", 
				tc.desc, nextValue, currentValue)
		}
	}
}

// TestMomentumLearning tests the momentum learning of the network
func TestMomentumLearning(t *testing.T) {
	network := NewNetwork()
	
	// Test consistent learning direction maintains momentum
	states := []env.State{
		{AngleRadians: 0.4, AngularVel: 0.5},
		{AngleRadians: 0.3, AngularVel: 0.4},
		{AngleRadians: 0.2, AngularVel: 0.3},
	}
	
	var prevWeights []float64
	var prevDelta float64
	
	for i, state := range states {
		initialWeights := network.GetWeights()
		network.Forward(state) // Force used internally for weight updates
		network.Update(0.8) // Consistent positive reward
		newWeights := network.GetWeights()
		
		// Calculate weight change magnitude
		var delta float64
		for j := range newWeights {
			delta += math.Abs(newWeights[j] - initialWeights[j])
		}
		
		if i > 0 {
			// Check if weight updates maintain direction
			sameDirection := true
			for j := range newWeights {
				if (newWeights[j] - initialWeights[j]) * (initialWeights[j] - prevWeights[j]) <= 0 {
					sameDirection = false
					break
				}
			}
			
			if !sameDirection {
				t.Error("Weight updates changed direction despite consistent rewards")
			}
			
			// Check if updates maintain or increase magnitude (momentum)
			if delta < prevDelta {
				t.Errorf("Update magnitude decreased: prev=%.4f, current=%.4f", prevDelta, delta)
			}
		}
		
		prevWeights = initialWeights
		prevDelta = delta
	}
}

// TestProgressiveLearningAdaptation verifies that the network adapts its learning
// based on state complexity
func TestProgressiveLearningAdaptation(t *testing.T) {
	// Given a network
	network := NewNetwork()
	network.SetDebug(false) // Disable debug output during tests
	
	// Define scenarios with increasing difficulty
	scenarios := []struct {
		name        string
		angleRange  float64 // Max angle deviation
		velRange    float64 // Max angular velocity
		targetForce float64 // Expected force direction
	}{
		{
			name:        "easy_balancing",
			angleRange:  0.1,  // Small angles
			velRange:    0.1,  // Slow movement
			targetForce: 0.0,  // Should learn to apply minimal force
		},
		{
			name:        "medium_recovery",
			angleRange:  0.3,  // Medium angles
			velRange:    0.5,  // Moderate movement
			targetForce: 2.0,  // Should learn to apply moderate force
		},
		{
			name:        "difficult_recovery",
			angleRange:  0.8,  // Large angles
			velRange:    1.5,  // Fast movement
			targetForce: 5.0,  // Should learn to apply maximum force
		},
	}
	
	// Track learning progress across scenarios
	var scenarioPerformance []float64
	
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// Train on this scenario
			const trainingSteps = 200
			var rewards []float64
			
			for i := 0; i < trainingSteps; i++ {
				// Generate random state within scenario parameters
				angle := (rand.Float64()*2-1) * scenario.angleRange
				velocity := (rand.Float64()*2-1) * scenario.velRange
				
				state := env.State{
					AngleRadians: angle,
					AngularVel:   velocity,
				}
				
				// Determine ideal force (simplified physics model)
				// For falling right (positive angle), apply negative force
				idealForce := -math.Copysign(
					math.Min(5.0, math.Abs(angle*10 + velocity*2)),
					angle,
				)
				
				// Get network's force
				force := network.Forward(state)
				
				// Calculate reward based on how close force is to ideal
				forceDiff := math.Abs(force - idealForce)
				reward := 1.0 - math.Min(1.0, forceDiff/10.0)
				
				// Update network
				network.Update(reward)
				rewards = append(rewards, reward)
			}
			
			// Calculate performance (average of last 50 rewards)
			lastRewards := rewards[max(0, len(rewards)-50):]
			avgReward := average(lastRewards)
			scenarioPerformance = append(scenarioPerformance, avgReward)
			
			// Log performance
			t.Logf("%s performance: %.4f", scenario.name, avgReward)
			
			// Verify learning occurred
			if avgReward < 0.6 {
				t.Errorf("%s: Network failed to achieve adequate performance: %.4f", 
					scenario.name, avgReward)
			}
		})
	}
	
	// Verify progressive learning (performance should be maintained or improve)
	for i := 1; i < len(scenarioPerformance); i++ {
		// Allow some regression (up to 20%) for harder scenarios
		if scenarioPerformance[i] < scenarioPerformance[i-1]*0.8 {
			t.Errorf("Progressive learning failed: performance dropped from %.4f to %.4f",
				scenarioPerformance[i-1], scenarioPerformance[i])
		}
	}
}

// Helper function to calculate average of a slice
func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// Helper function for max of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// TestMomentumBackpropagation tests the momentum backpropagation
func TestMomentumBackpropagation(t *testing.T) {
	network := NewNetwork()
	
	// Test sequence of updates with momentum
	states := []env.State{
		{AngleRadians: 0.2, AngularVel: 0.3},
		{AngleRadians: 0.15, AngularVel: 0.2},
		{AngleRadians: 0.1, AngularVel: 0.1},
	}
	
	var prevDelta float64
	for i, state := range states {
		force := network.Forward(state)
		delta := network.Update(0.8) // Positive reward for improvement
		
		if i > 0 {
			// Momentum should cause larger updates in consistent direction
			if math.Abs(delta) <= math.Abs(prevDelta) {
				t.Errorf("Momentum not increasing update magnitude: prev=%v, current=%v", prevDelta, delta)
			}
		}
		prevDelta = delta
	}
}

// TestAdaptiveLearningRate tests the adaptive learning rate
func TestAdaptiveLearningRate(t *testing.T) {
	network := NewNetwork()
	successRates := []float64{0.2, 0.4, 0.6, 0.8}
	
	var prevLearningRate float64
	for _, rate := range successRates {
		network.UpdateSuccessRate(rate)
		currentRate := network.GetLearningRate()
		
		if rate > 0.5 && currentRate <= prevLearningRate {
			t.Errorf("Learning rate should increase with high success: prev=%v, current=%v", prevLearningRate, currentRate)
		}
		prevLearningRate = currentRate
	}
}

// TestStateTransitionQuality tests the state transition quality
func TestStateTransitionQuality(t *testing.T) {
	network := NewNetwork()
	
	transitions := []struct {
		initial env.State
		action float64
		final  env.State
		expectedQuality float64
	}{
		{
			initial: env.State{AngleRadians: 0.3, AngularVel: 0.4},
			action: -2.0,
			final: env.State{AngleRadians: 0.2, AngularVel: 0.3},
			expectedQuality: 0.8,
		},
		{
			initial: env.State{AngleRadians: 0.2, AngularVel: 0.3},
			action: -1.5,
			final: env.State{AngleRadians: 0.1, AngularVel: 0.2},
			expectedQuality: 0.9,
		},
	}
	
	for _, tr := range transitions {
		quality := network.EvaluateTransition(tr.initial, tr.action, tr.final)
		if math.Abs(quality - tr.expectedQuality) > 0.2 {
			t.Errorf("Transition quality mismatch: got=%v, want=%v", quality, tr.expectedQuality)
		}
	}
}

// TestLongTermStability tests the long-term stability
func TestLongTermStability(t *testing.T) {
	network := NewNetwork()
	const episodes = 100
	
	var totalReward float64
	var prevAvgReward float64
	
	for i := 0; i < episodes; i++ {
		state := env.State{
			AngleRadians: 0.1 * math.Sin(float64(i)),
			AngularVel: 0.1 * math.Cos(float64(i)),
		}
		
		force := network.Forward(state)
		reward := -math.Abs(state.AngleRadians) // Reward staying vertical
		network.Update(reward)
		
		totalReward += reward
		avgReward := totalReward / float64(i+1)
		
		if i > episodes/2 && avgReward < prevAvgReward {
			t.Errorf("Learning stability degraded: prev_avg=%v, current_avg=%v", prevAvgReward, avgReward)
		}
		prevAvgReward = avgReward
	}
}

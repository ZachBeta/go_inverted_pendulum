package neural

import (
	"testing"
	"math"
	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
)

func TestProgressiveLearningBehavior(t *testing.T) {
	network := NewNetwork()
	network.SetLearningRate(0.1) // Higher learning rate for more noticeable changes
	
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
	for _, tc := range states {
		// Get network's prediction before update
		initialValue := network.Predict(tc.state.AngleRadians, tc.state.AngularVel)
		
		// Perform multiple updates to allow learning to take effect
		for i := 0; i < 3; i++ {
			_ = network.Forward(tc.state)
			network.Update(tc.reward)
		}
		
		// Get prediction after update
		newValue := network.Predict(tc.state.AngleRadians, tc.state.AngularVel)
		
		// Verify learning occurred - allow for both positive and negative changes
		// as long as there is some significant change
		if math.Abs(newValue-initialValue) < 0.001 {
			t.Errorf("%s: No significant learning occurred (before: %.4f, after: %.4f)", 
				tc.desc, initialValue, newValue)
		}
	}
}

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

func TestMomentumLearning(t *testing.T) {
	network := NewNetwork()
	network.SetLearningRate(0.1) // Set higher learning rate for momentum test
	
	// Test consistent learning direction maintains momentum
	states := []env.State{
		{AngleRadians: 0.4, AngularVel: 0.5},
		{AngleRadians: 0.3, AngularVel: 0.4},
		{AngleRadians: 0.2, AngularVel: 0.3},
	}
	
	var prevWeights []float64
	var prevDelta float64
	
	for _, state := range states {
		initialWeights := network.GetWeights()
		_ = network.Forward(state) // Force used internally for weight updates
		network.Update(0.8)
		newWeights := network.GetWeights()
		
		// Calculate weight change magnitude
		var delta float64
		for j := range newWeights {
			delta += math.Abs(newWeights[j] - initialWeights[j])
		}
		
		if prevWeights != nil {
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
			
			// Allow for small variations in update magnitude
			if delta < prevDelta*0.5 {
				t.Errorf("Update magnitude decreased too much: prev=%.4f, current=%.4f", prevDelta, delta)
			}
		}
		
		prevWeights = initialWeights
		prevDelta = delta
	}
}

func TestLearningRateAdaptation(t *testing.T) {
	network := NewNetwork()
	initialRate := network.GetLearningRate()
	
	// Train with consistently good performance
	state := env.State{AngleRadians: 0.1, AngularVel: 0.1}
	
	// Multiple updates with good rewards
	for i := 0; i < 5; i++ {
		_ = network.Forward(state)
		network.Update(0.9) // High reward
	}
	
	adaptedRate := network.GetLearningRate()
	
	// Check for any adaptation (could increase or decrease)
	if math.Abs(adaptedRate-initialRate) < 0.0001 {
		t.Error("Learning rate showed no adaptation despite consistent performance")
	}
}

func TestStateTransitionLearning(t *testing.T) {
	network := NewNetwork()
	network.SetLearningRate(0.1) // Higher learning rate for more noticeable changes
	
	// Test sequence of state transitions
	transitions := []struct {
		current env.State
		next    env.State
		reward  float64
	}{
		{
			current: env.State{AngleRadians: 0.3, AngularVel: 0.4},
			next:    env.State{AngleRadians: 0.2, AngularVel: 0.3},
			reward:  0.7,
		},
		{
			current: env.State{AngleRadians: 0.2, AngularVel: 0.3},
			next:    env.State{AngleRadians: 0.1, AngularVel: 0.2},
			reward:  0.8,
		},
	}
	
	for _, tc := range transitions {
		// Get initial prediction
		initialValue := network.Predict(tc.current.AngleRadians, tc.current.AngularVel)
		
		// Multiple updates to allow learning to take effect
		for i := 0; i < 3; i++ {
			_ = network.Forward(tc.current)
			network.Update(tc.reward)
		}
		
		// Get updated prediction
		newValue := network.Predict(tc.current.AngleRadians, tc.current.AngularVel)
		
		// Verify learning from transition - allow for both positive and negative changes
		if math.Abs(newValue-initialValue) < 0.001 {
			t.Errorf("Network showed no significant learning from transition (before: %.4f, after: %.4f)",
				initialValue, newValue)
		}
	}
}

func TestProgressiveTraining(t *testing.T) {
	network := NewNetwork()
	
	// Initial state check
	if network.GetDifficulty() != 0.1 {
		t.Errorf("Initial difficulty should be 0.1, got %.2f", network.GetDifficulty())
	}
	if network.GetSuccessRate() != 0.0 {
		t.Errorf("Initial success rate should be 0.0, got %.2f", network.GetSuccessRate())
	}
	
	// Test difficulty increase with high success
	for i := 0; i < 80; i++ { // 80% success rate
		_ = network.Forward(env.State{AngleRadians: 0.1, AngularVel: 0.1})
		network.Update(0.9) // High reward = success
	}
	for i := 0; i < 20; i++ { // 20% failure
		_ = network.Forward(env.State{AngleRadians: 0.3, AngularVel: 0.3})
		network.Update(-0.1) // Low reward = failure
	}
	
	// Check difficulty increased
	newDiff := network.GetDifficulty()
	if newDiff <= 0.1 {
		t.Errorf("Difficulty should have increased from 0.1, got %.2f", newDiff)
	}
	
	// Success rate should be around 0.8
	successRate := network.GetSuccessRate()
	if math.Abs(successRate - 0.8) > 0.1 {
		t.Errorf("Success rate should be close to 0.8, got %.2f", successRate)
	}
	
	// Test difficulty decrease with low success
	network = NewNetwork() // Reset for clean test
	
	for i := 0; i < 20; i++ { // 20% success
		_ = network.Forward(env.State{AngleRadians: 0.1, AngularVel: 0.1})
		network.Update(0.9)
	}
	for i := 0; i < 80; i++ { // 80% failure
		_ = network.Forward(env.State{AngleRadians: 0.5, AngularVel: 0.5})
		network.Update(-0.1)
	}
	
	// Check difficulty decreased
	finalDiff := network.GetDifficulty()
	if finalDiff >= 0.1 {
		t.Errorf("Difficulty should have decreased from 0.1, got %.2f", finalDiff)
	}
}

func TestDifficultyBasedLearning(t *testing.T) {
	network := NewNetwork()
	
	// Test states at different difficulties
	tests := []struct {
		difficulty float64
		state      env.State
		reward     float64
	}{
		{0.1, env.State{AngleRadians: 0.1, AngularVel: 0.1}, 0.9},  // Easy
		{0.5, env.State{AngleRadians: 0.3, AngularVel: 0.3}, 0.7},  // Medium
		{1.0, env.State{AngleRadians: 0.5, AngularVel: 0.5}, 0.5},  // Hard
	}
	
	for _, tc := range tests {
		// Force specific difficulty for test
		network = NewNetwork()
		for network.GetDifficulty() < tc.difficulty {
			network.Update(0.9) // Force difficulty increase
		}
		
		// Get initial prediction
		initialValue := network.Predict(tc.state.AngleRadians, tc.state.AngularVel)
		
		// Perform learning step
		_ = network.Forward(tc.state)
		network.Update(tc.reward)
		
		// Get new prediction
		newValue := network.Predict(tc.state.AngleRadians, tc.state.AngularVel)
		
		// Higher difficulty should lead to larger learning steps
		learningMagnitude := math.Abs(newValue - initialValue)
		
		if tc.difficulty > 0.5 && learningMagnitude < 0.01 {
			t.Errorf("Learning magnitude %.4f too small for difficulty %.2f", 
				learningMagnitude, tc.difficulty)
		}
		
		if tc.difficulty < 0.2 && learningMagnitude > 0.1 {
			t.Errorf("Learning magnitude %.4f too large for difficulty %.2f", 
				learningMagnitude, tc.difficulty)
		}
	}
}

func TestSuccessRateTracking(t *testing.T) {
	network := NewNetwork()
	
	// Test success rate calculation accuracy
	successPatterns := []struct {
		rewards []float64
		wantRate float64
		desc string
	}{
		{
			rewards: []float64{0.9, 0.9, 0.9, 0.9, -0.1}, // 80% success
			wantRate: 0.8,
			desc: "High success rate",
		},
		{
			rewards: []float64{-0.1, -0.1, -0.1, 0.9, 0.9}, // 40% success
			wantRate: 0.4,
			desc: "Medium success rate",
		},
		{
			rewards: []float64{-0.1, -0.1, -0.1, -0.1, 0.9}, // 20% success
			wantRate: 0.2,
			desc: "Low success rate",
		},
	}
	
	for _, tc := range successPatterns {
		network = NewNetwork() // Reset for each test
		
		// Apply rewards
		for _, reward := range tc.rewards {
			_ = network.Forward(env.State{AngleRadians: 0.1, AngularVel: 0.1})
			network.Update(reward)
		}
		
		// Check success rate
		rate := network.GetSuccessRate()
		if math.Abs(rate - tc.wantRate) > 0.01 {
			t.Errorf("%s: got success rate %.2f, want %.2f", 
				tc.desc, rate, tc.wantRate)
		}
	}
}

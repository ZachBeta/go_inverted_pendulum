package training

import (
	"bytes"
	"encoding/json"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/neural"
)

func TestTrainingProgress(t *testing.T) {
	// Create test network and trainer
	network := neural.NewNetwork()
	var logBuf bytes.Buffer
	logger := log.New(&logBuf, "", 0)
	config := NewDefaultConfig()
	
	scenarios := []struct {
		name     string
		state    env.State
		action   float64
		reward   float64
		wantLog  string
	}{
		{
			name: "initial_learning",
			state: env.State{
				AngleRadians: 0.1,  // Nearly vertical
				AngularVel:   0.05, // Slow movement
			},
			action: 0.5,
			reward: 0.9,
			wantLog: "Average Reward: 0.9",
		},
		{
			name: "recovery_learning",
			state: env.State{
				AngleRadians: math.Pi / 4, // 45 degrees
				AngularVel:   1.0,         // Fast movement
			},
			action: -2.0,
			reward: -0.5,
			wantLog: "Average Reward: -0.5",
		},
		{
			name: "stability_learning",
			state: env.State{
				AngleRadians: math.Pi / 12, // 15 degrees
				AngularVel:   0.2,          // Moderate movement
			},
			action: 1.0,
			reward: 0.7,
			wantLog: "Average Reward: 0.7",
		},
	}

	for _, tc := range scenarios {
		t.Run(tc.name, func(t *testing.T) {
			// Create new trainer for each test case
			trainer := NewTrainer(config, network, logger)
			logBuf.Reset()
			
			// Add experience and process batch
			trainer.AddExperience(Experience{
				State:  tc.state,
				Action: tc.action,
				Reward: tc.reward,
			})
			
			// Force batch processing
			trainer.processBatch()
			
			// Verify logging output
			if !strings.Contains(logBuf.String(), tc.wantLog) {
				t.Errorf("processBatch() log = %v, want containing %v", logBuf.String(), tc.wantLog)
			}
			
			// Verify metrics collection
			stats := trainer.GetTrainingStats()
			metrics := stats["metrics"].(*MetricsCollector)
			
			if metrics.ExperienceCount != 1 {
				t.Errorf("metrics.ExperienceCount = %v, want 1", metrics.ExperienceCount)
			}
			
			if metrics.TotalReward != tc.reward {
				t.Errorf("metrics.TotalReward = %v, want %v", metrics.TotalReward, tc.reward)
			}

			// Verify angle tracking
			if metrics.MaxAngle < math.Abs(tc.state.AngleRadians) {
				t.Errorf("MaxAngle = %v, want >= %v", metrics.MaxAngle, math.Abs(tc.state.AngleRadians))
			}
		})
	}
}

func TestTemporalDifferenceLearning(t *testing.T) {
	network := neural.NewNetwork()
	config := NewDefaultConfig()
	config.BatchSize = 2 // Process pairs of experiences
	trainer := NewTrainer(config, network, nil)

	scenarios := []struct {
		name          string
		experiences   []Experience
		wantImproved bool
		description  string
	}{
		{
			name: "improving_sequence",
			experiences: []Experience{
				{
					State: env.State{AngleRadians: 0.5, AngularVel: 1.0},
					Action: -2.0,
					Reward: -0.5,
				},
				{
					State: env.State{AngleRadians: 0.3, AngularVel: 0.5},
					Action: -1.0,
					Reward: -0.2,
				},
			},
			wantImproved: true,
			description:  "Value should increase as state improves",
		},
		{
			name: "worsening_sequence",
			experiences: []Experience{
				{
					State: env.State{AngleRadians: 0.1, AngularVel: 0.2},
					Action: 1.0,
					Reward: 0.8,
				},
				{
					State: env.State{AngleRadians: 0.3, AngularVel: 0.6},
					Action: 2.0,
					Reward: -0.3,
				},
			},
			wantImproved: false,
			description:  "Value should decrease as state worsens",
		},
	}

	for _, tc := range scenarios {
		t.Run(tc.name, func(t *testing.T) {
			// Add experiences
			for _, exp := range tc.experiences {
				trainer.AddExperience(exp)
			}

			// Get initial state values
			initialValue := network.Predict(
				tc.experiences[0].State.AngleRadians,
				tc.experiences[0].State.AngularVel,
			)

			// Process batch to apply TD learning
			trainer.processBatch()

			// Get final state values
			finalValue := network.Predict(
				tc.experiences[len(tc.experiences)-1].State.AngleRadians,
				tc.experiences[len(tc.experiences)-1].State.AngularVel,
			)

			if tc.wantImproved && finalValue <= initialValue {
				t.Errorf("%s: value did not improve as expected: initial=%.4f, final=%.4f",
					tc.description, initialValue, finalValue)
			} else if !tc.wantImproved && finalValue >= initialValue {
				t.Errorf("%s: value did not decrease as expected: initial=%.4f, final=%.4f",
					tc.description, initialValue, finalValue)
			}
		})
	}
}

func TestProgressiveLearning(t *testing.T) {
	network := neural.NewNetwork()
	config := NewDefaultConfig()
	config.BatchSize = 1 // Process each experience immediately for testing
	trainer := NewTrainer(config, network, nil)

	// Test progressive improvement
	states := []env.State{
		{AngleRadians: math.Pi / 2, AngularVel: 1.0},  // Start poorly (90 degrees)
		{AngleRadians: math.Pi / 4, AngularVel: 0.5},  // Improve (45 degrees)
		{AngleRadians: math.Pi / 6, AngularVel: 0.2},  // Better (30 degrees)
		{AngleRadians: math.Pi / 12, AngularVel: 0.1}, // Good (15 degrees)
	}

	var lastReward float64
	for i, state := range states {
		trainer.AddExperience(Experience{
			State:  state,
			Action: -state.AngularVel, // Simple corrective action
			Reward: 1.0 - math.Abs(state.AngleRadians)/math.Pi,
		})

		stats := trainer.GetTrainingStats()
		metrics := stats["metrics"].(*MetricsCollector)
		currentReward := metrics.TotalReward

		if i > 0 && currentReward <= lastReward {
			t.Errorf("Reward not improving: current=%v, previous=%v", currentReward, lastReward)
		}
		lastReward = currentReward
	}
}

func TestBatchProcessing(t *testing.T) {
	network := neural.NewNetwork()
	config := NewDefaultConfig()
	config.BatchSize = 3
	trainer := NewTrainer(config, network, nil)

	// Add experiences just under batch size
	for i := 0; i < config.BatchSize-1; i++ {
		trainer.AddExperience(Experience{
			State: env.State{
				AngleRadians: float64(i) * 0.1,
				AngularVel:   float64(i) * 0.2,
			},
			Action: float64(i),
			Reward: 1.0,
		})
	}

	// Verify batch not processed yet
	stats := trainer.GetTrainingStats()
	metrics := stats["metrics"].(*MetricsCollector)
	beforeProcessed := metrics.BatchCount

	// Add one more experience to trigger batch processing
	trainer.AddExperience(Experience{
		State: env.State{
			AngleRadians: 0.5,
			AngularVel:   1.0,
		},
		Action: 2.0,
		Reward: 1.0,
	})

	// Verify batch was processed
	stats = trainer.GetTrainingStats()
	metrics = stats["metrics"].(*MetricsCollector)
	afterProcessed := metrics.BatchCount

	if afterProcessed <= beforeProcessed {
		t.Errorf("Batch not processed when size threshold reached: before=%d, after=%d",
			beforeProcessed, afterProcessed)
	}
}

func TestCheckpointSaving(t *testing.T) {
	// Create temporary directory for checkpoints
	tmpDir, err := os.MkdirTemp("", "training_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// Initialize trainer with custom checkpoint directory
	network := neural.NewNetwork()
	config := NewDefaultConfig()
	config.CheckpointInterval = 1 // Save every episode
	trainer := NewTrainer(config, network, nil)
	trainer.SetCheckpointDirectory(tmpDir)

	// Add experiences with progressive improvement
	angles := []float64{math.Pi / 3, math.Pi / 4, math.Pi / 6}
	for _, angle := range angles {
		trainer.AddExperience(Experience{
			State: env.State{
				AngleRadians: angle,
				AngularVel:   angle / 2,
			},
			Action: -angle,
			Reward: 1.0 - math.Abs(angle)/math.Pi,
		})
	}

	// End episode to trigger checkpoint
	trainer.OnEpisodeEnd(100)

	// Verify checkpoint files exist
	weightsFile := filepath.Join(tmpDir, "weights_episode_0.json")
	metricsFile := filepath.Join(tmpDir, "metrics_episode_0.json")

	if _, err := os.Stat(weightsFile); os.IsNotExist(err) {
		t.Error("weights checkpoint file not created")
	}
	if _, err := os.Stat(metricsFile); os.IsNotExist(err) {
		t.Error("metrics checkpoint file not created")
	}

	// Verify checkpoint contents and structure
	files := []string{weightsFile, metricsFile}
	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Errorf("failed to read checkpoint file %s: %v", file, err)
			continue
		}

		if len(data) == 0 {
			t.Errorf("checkpoint file %s is empty", file)
		}

		// Verify JSON structure
		var checkpoint map[string]interface{}
		if err := json.Unmarshal(data, &checkpoint); err != nil {
			t.Errorf("invalid JSON in checkpoint file %s: %v", file, err)
			continue
		}

		// Check required fields based on file type
		if strings.Contains(filepath.Base(file), "weights") {
			if _, ok := checkpoint["weights"]; !ok {
				t.Errorf("weights file %s missing weights field", file)
			}
			if _, ok := checkpoint["timestamp"]; !ok {
				t.Errorf("weights file %s missing timestamp field", file)
			}
		} else if strings.Contains(filepath.Base(file), "metrics") {
			if _, ok := checkpoint["metrics"]; !ok {
				t.Errorf("metrics file %s missing metrics field", file)
			}
			if _, ok := checkpoint["episode"]; !ok {
				t.Errorf("metrics file %s missing episode field", file)
			}
		}
	}
}

func TestCheckpointRestoration(t *testing.T) {
	// Create temporary directory for checkpoints
	tmpDir, err := os.MkdirTemp("", "training_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// Create initial trainer and save checkpoint
	network := neural.NewNetwork()
	config := NewDefaultConfig()
	trainer := NewTrainer(config, network, nil)

	// Add some experiences
	trainer.AddExperience(Experience{
		State: env.State{AngleRadians: 0.1, AngularVel: 0.2},
		Action: -0.5,
		Reward: 0.8,
	})

	// Save checkpoint
	trainer.SetCheckpointDirectory(tmpDir)
	trainer.saveCheckpoint()

	// Create new trainer and restore from checkpoint
	newNetwork := neural.NewNetwork()
	newTrainer := NewTrainer(config, newNetwork, nil)

	// Load weights and metrics
	weightsFile := filepath.Join(tmpDir, "weights_episode_0.json")
	metricsFile := filepath.Join(tmpDir, "metrics_episode_0.json")
	
	if err := newTrainer.LoadCheckpoint(weightsFile); err != nil {
		t.Fatalf("failed to load weights checkpoint: %v", err)
	}

	// Load metrics
	data, err := os.ReadFile(metricsFile)
	if err != nil {
		t.Fatalf("failed to read metrics file: %v", err)
	}

	var metricsData struct {
		Episode int               `json:"episode"`
		Metrics *MetricsCollector `json:"metrics"`
	}
	if err := json.Unmarshal(data, &metricsData); err != nil {
		t.Fatalf("failed to unmarshal metrics: %v", err)
	}
	newTrainer.metrics = metricsData.Metrics

	// Compare metrics between trainers
	origStats := trainer.GetTrainingStats()
	newStats := newTrainer.GetTrainingStats()

	origMetrics := origStats["metrics"].(*MetricsCollector)
	newMetrics := newStats["metrics"].(*MetricsCollector)

	if origMetrics.ExperienceCount != newMetrics.ExperienceCount {
		t.Errorf("experience count mismatch after restore: orig=%d, new=%d",
			origMetrics.ExperienceCount, newMetrics.ExperienceCount)
	}

	if origMetrics.TotalReward != newMetrics.TotalReward {
		t.Errorf("total reward mismatch after restore: orig=%f, new=%f",
			origMetrics.TotalReward, newMetrics.TotalReward)
	}
}

func TestMetricsCollection(t *testing.T) {
	collector := NewMetricsCollector(1)

	// Test progressive learning metrics
	experiences := []struct {
		angle  float64
		vel    float64
		reward float64
	}{
		{math.Pi / 2, 1.0, 0.2},   // Poor performance
		{math.Pi / 4, 0.5, 0.5},   // Improving
		{math.Pi / 6, 0.2, 0.8},   // Good
		{math.Pi / 12, 0.1, 0.95}, // Excellent
	}

	for _, exp := range experiences {
		collector.RecordExperience(Experience{
			State:  env.State{AngleRadians: exp.angle, AngularVel: exp.vel},
			Reward: exp.reward,
		})
	}

	// Verify metrics
	if collector.ExperienceCount != len(experiences) {
		t.Errorf("ExperienceCount = %v, want %v", collector.ExperienceCount, len(experiences))
	}

	expectedTotalReward := 0.0
	for _, exp := range experiences {
		expectedTotalReward += exp.reward
	}
	if math.Abs(collector.TotalReward-expectedTotalReward) > 1e-6 {
		t.Errorf("TotalReward = %v, want %v", collector.TotalReward, expectedTotalReward)
	}

	if collector.MaxAngle != math.Pi/2 {
		t.Errorf("MaxAngle = %v, want %v", collector.MaxAngle, math.Pi/2)
	}

	if collector.MinAngle != math.Pi/12 {
		t.Errorf("MinAngle = %v, want %v", collector.MinAngle, math.Pi/12)
	}

	// Test weight updates
	weights := []struct{ angle, vel, bias float64 }{
		{2.0, 1.0, 0.1},
		{2.1, 1.1, 0.15},
		{2.2, 1.2, 0.18},
	}

	for _, w := range weights {
		collector.RecordWeightUpdate(w.angle, w.vel, w.bias)
	}

	if len(collector.WeightUpdates) != len(weights) {
		t.Errorf("WeightUpdates count = %v, want %v", len(collector.WeightUpdates), len(weights))
	}

	// Verify metrics output format
	str := collector.String()
	expectedSubstrings := []string{
		"Episode 1",
		"Experiences: 4",
		"Max Angle:",
		"Min Angle:",
		"Avg Reward:",
	}

	for _, substr := range expectedSubstrings {
		if !strings.Contains(str, substr) {
			t.Errorf("String() = %v, want containing %v", str, substr)
		}
	}
}

package training

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/zachbeta/go_inverted_pendulum/pkg/neural"
)

// Trainer manages the training process for the neural network
type Trainer struct {
	config         Config
	network        *neural.Network
	logger         *log.Logger
	batch          *Batch
	metrics        *MetricsCollector
	episode        int
	learningRate   float64
	totalEpisodes  int
	successCount   int     // Episodes where pendulum stayed upright
	bestDuration   float64 // Best upright duration in seconds
	checkpointDir  string  // Directory for saving checkpoints
	lastCheckpoint time.Time // Time of last checkpoint save
}

// NewTrainer creates a new trainer with the given config
func NewTrainer(config Config, network *neural.Network, logger *log.Logger) *Trainer {
	if logger == nil {
		logger = log.Default()
	}

	return &Trainer{
		config:        config,
		network:      network,
		logger:       logger,
		batch:        &Batch{Experiences: make([]Experience, 0, config.BatchSize)},
		metrics:      NewMetricsCollector(0),
		episode:      0,
		learningRate: config.BaseLearningRate,
		checkpointDir: "checkpoints", // Default directory
		lastCheckpoint: time.Now(),
	}
}

// SetCheckpointDirectory sets the directory for saving checkpoints
func (t *Trainer) SetCheckpointDirectory(dir string) {
	t.checkpointDir = dir
}

// AddExperience adds a new experience to the current batch
func (t *Trainer) AddExperience(exp Experience) {
	// Record metrics
	t.metrics.RecordExperience(exp)

	// Initialize batch if needed
	if len(t.batch.Experiences) == 0 {
		t.batch.StartTime = time.Now()
		t.batch.EpisodeID = t.episode
	}

	t.batch.Experiences = append(t.batch.Experiences, exp)

	// Process batch if full
	if len(t.batch.Experiences) >= t.config.BatchSize {
		t.processBatch()
	}
}

// processBatch applies the batch update to the network weights using backpropagation
func (t *Trainer) processBatch() {
	if len(t.batch.Experiences) == 0 {
		return
	}

	t.batch.EndTime = time.Now()

	// Calculate average reward and gradients for the batch
	var totalReward float64
	angleGrad := 0.0
	angularVelGrad := 0.0
	biasGrad := 0.0

	// Progressive learning: Focus on experiences with better rewards
	sortExperiencesByReward(t.batch.Experiences)
	effectiveBatchSize := int(float64(len(t.batch.Experiences)) * 0.8) // Use top 80% of experiences
	if effectiveBatchSize < 1 {
		effectiveBatchSize = 1
	}

	// Apply backpropagation with momentum
	momentum := 0.9
	prevAngleGrad := 0.0
	prevAngularVelGrad := 0.0
	prevBiasGrad := 0.0

	for i := 0; i < effectiveBatchSize; i++ {
		exp := t.batch.Experiences[i]
		actionSign := sign(exp.Action)

		// Calculate gradients with temporal difference
		nextValue := t.network.Predict(exp.NextState.AngleRadians, exp.NextState.AngularVel)
		currentValue := t.network.Predict(exp.State.AngleRadians, exp.State.AngularVel)
		tdError := exp.Reward + 0.99*nextValue - currentValue // 0.99 is discount factor

		// Compute gradients with momentum
		angleGrad = momentum*prevAngleGrad + (1-momentum)*tdError*exp.State.AngleRadians*actionSign
		angularVelGrad = momentum*prevAngularVelGrad + (1-momentum)*tdError*exp.State.AngularVel*actionSign
		biasGrad = momentum*prevBiasGrad + (1-momentum)*tdError*actionSign

		prevAngleGrad = angleGrad
		prevAngularVelGrad = angularVelGrad
		prevBiasGrad = biasGrad

		totalReward += exp.Reward
	}

	// Average the gradients
	batchSize := float64(effectiveBatchSize)
	angleGrad /= batchSize
	angularVelGrad /= batchSize
	biasGrad /= batchSize

	// Apply gradients with adaptive learning rate
	weights := t.network.GetWeights()
	newWeights := []float64{
		clip(weights[0]+t.learningRate*angleGrad, t.config.WeightClipMin, t.config.WeightClipMax),
		clip(weights[1]+t.learningRate*angularVelGrad, t.config.WeightClipMin, t.config.WeightClipMax),
		clip(weights[2]+t.learningRate*biasGrad, t.config.WeightClipMin, t.config.WeightClipMax),
	}

	// Update network weights and record metrics
	t.network.SetWeights(newWeights)
	t.metrics.RecordWeightUpdate(newWeights[0], newWeights[1], newWeights[2])
	t.metrics.RecordBatchProcessed()

	// Log batch results with clear formatting
	t.logger.Printf("\n[Trainer] Batch Update Summary:")
	t.logger.Printf("├── Episode: %d", t.episode)
	t.logger.Printf("├── Batch Size: %d (effective: %d)", len(t.batch.Experiences), effectiveBatchSize)
	t.logger.Printf("├── Average Reward: %.4f", totalReward/batchSize)
	t.logger.Printf("├── Learning Rate: %.4f", t.learningRate)
	t.logger.Printf("├── Weight Updates")
	t.logger.Printf("│   ├── Angle: %.4f", angleGrad)
	t.logger.Printf("│   ├── Angular Velocity: %.4f", angularVelGrad)
	t.logger.Printf("│   └── Bias: %.4f", biasGrad)
	t.logger.Printf("└── New Weights")
	t.logger.Printf("    ├── Angle: %.4f", newWeights[0])
	t.logger.Printf("    ├── Angular Velocity: %.4f", newWeights[1])
	t.logger.Printf("    └── Bias: %.4f", newWeights[2])

	// Reset batch
	t.batch.Experiences = t.batch.Experiences[:0]
}

// OnEpisodeEnd handles end-of-episode processing
func (t *Trainer) OnEpisodeEnd(episodeTicks int) {
	// Process any remaining experiences in the batch
	t.processBatch()

	// Calculate episode success metrics
	duration := float64(episodeTicks) * t.config.DeltaTime
	if t.metrics.MaxAngle < t.config.SuccessAngleThresh {
		t.successCount++
		if duration > t.bestDuration {
			t.bestDuration = duration
			t.logger.Printf("\n[Trainer] New Best Duration: %.2f seconds!", duration)
		}
	}

	// Log episode summary
	t.logger.Printf("\n%s", t.metrics.String())
	if t.successCount > 0 {
		t.logger.Printf("Success Rate: %.1f%% (%d/%d episodes)",
			100*float64(t.successCount)/float64(t.totalEpisodes+1),
			t.successCount, t.totalEpisodes+1)
	}

	// Adaptive learning rate with success-based adjustment
	successRate := float64(t.successCount) / float64(t.totalEpisodes+1)
	if successRate > 0.7 {
		// Reduce learning rate more aggressively when performing well
		t.learningRate *= t.config.LearningRateDecay * 0.9
	} else {
		t.learningRate *= t.config.LearningRateDecay
	}
	t.learningRate = math.Max(t.config.MinLearningRate, t.learningRate)

	// Save checkpoint if needed
	if t.episode%t.config.CheckpointInterval == 0 || 
	   time.Since(t.lastCheckpoint) > 5*time.Minute {
		t.saveCheckpoint()
		t.lastCheckpoint = time.Now()
	}

	// Update episode counters and reset metrics
	t.episode++
	t.totalEpisodes++
	t.metrics = NewMetricsCollector(t.episode)
}

// GetTrainingStats returns current training statistics
func (t *Trainer) GetTrainingStats() map[string]interface{} {
	return map[string]interface{}{
		"episode":        t.episode,
		"totalEpisodes": t.totalEpisodes,
		"successCount":  t.successCount,
		"bestDuration":  t.bestDuration,
		"learningRate":  t.learningRate,
		"metrics":       t.metrics,
	}
}

// saveCheckpoint saves the current network state and training metrics
func (t *Trainer) saveCheckpoint() {
	// Create checkpoint directory if it doesn't exist
	if err := os.MkdirAll(t.checkpointDir, 0755); err != nil {
		t.logger.Printf("Failed to create checkpoint directory: %v", err)
		return
	}

	// Save network weights
	weights := t.network.GetWeights()
	weightsCheckpoint := filepath.Join(t.checkpointDir, fmt.Sprintf("weights_episode_%d.json", t.episode))
	weightsData := map[string]interface{}{
		"episode":    t.episode,
		"weights":    weights,
		"timestamp": time.Now(),
	}
	if data, err := json.MarshalIndent(weightsData, "", "  "); err == nil {
		if err := os.WriteFile(weightsCheckpoint, data, 0644); err != nil {
			t.logger.Printf("Failed to save weights checkpoint: %v", err)
		}
	}

	// Save metrics
	metricsCheckpoint := filepath.Join(t.checkpointDir, fmt.Sprintf("metrics_episode_%d.json", t.episode))
	metricsData := map[string]interface{}{
		"episode":    t.episode,
		"metrics":    t.metrics,
		"timestamp": time.Now(),
	}
	if data, err := json.MarshalIndent(metricsData, "", "  "); err == nil {
		if err := os.WriteFile(metricsCheckpoint, data, 0644); err != nil {
			t.logger.Printf("Failed to save metrics checkpoint: %v", err)
		}
	}

	t.logger.Printf("[Trainer] Saved checkpoint to %s", weightsCheckpoint)
}

// LoadCheckpoint loads a checkpoint from the given file
func (t *Trainer) LoadCheckpoint(path string) error {
	// Read checkpoint file
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read checkpoint: %w", err)
	}

	// Parse checkpoint data
	var checkpoint struct {
		Episode    int       `json:"episode"`
		Weights    []float64 `json:"weights"`
		Timestamp  time.Time `json:"timestamp"`
	}
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return fmt.Errorf("failed to unmarshal checkpoint: %w", err)
	}

	// Restore network state
	if err := t.network.SetWeights(checkpoint.Weights); err != nil {
		return fmt.Errorf("failed to restore weights: %w", err)
	}

	// Update trainer state
	t.episode = checkpoint.Episode
	t.metrics = NewMetricsCollector(t.episode)

	return nil
}

// sortExperiencesByReward sorts experiences by reward in descending order
func sortExperiencesByReward(experiences []Experience) {
	// Simple bubble sort since batch sizes are small
	n := len(experiences)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if experiences[j].Reward < experiences[j+1].Reward {
				experiences[j], experiences[j+1] = experiences[j+1], experiences[j]
			}
		}
	}
}

// sign returns the sign of a number: 1 for positive, -1 for negative, 0 for zero
func sign(x float64) float64 {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}

// clip limits a value to [min, max] range
func clip(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

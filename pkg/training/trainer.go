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
	config     Config
	network    *neural.Network
	logger     *log.Logger
	batch      *Batch
	metrics    []Metrics
	episode    int
	learningRate float64
}

// NewTrainer creates a new trainer with the given config
func NewTrainer(config Config, network *neural.Network, logger *log.Logger) *Trainer {
	if logger == nil {
		logger = log.Default()
	}

	return &Trainer{
		config:       config,
		network:      network,
		logger:       logger,
		batch:        &Batch{Experiences: make([]Experience, 0, config.BatchSize)},
		metrics:      make([]Metrics, 0),
		episode:      0,
		learningRate: config.BaseLearningRate,
	}
}

// AddExperience adds a new experience to the current batch
func (t *Trainer) AddExperience(exp Experience) {
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

// processBatch applies the batch update to the network weights
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

	for _, exp := range t.batch.Experiences {
		// Get the sign of the action for consistent updates
		actionSign := sign(exp.Action)

		// Calculate gradients based on the reward signal
		angleGrad += exp.Reward * exp.State.AngleRadians * actionSign
		angularVelGrad += exp.Reward * exp.State.AngularVel * actionSign
		biasGrad += exp.Reward * actionSign
		totalReward += exp.Reward
	}

	// Average the gradients
	batchSize := float64(len(t.batch.Experiences))
	angleGrad /= batchSize
	angularVelGrad /= batchSize
	biasGrad /= batchSize

	// Apply gradients with current learning rate
	weights := t.network.GetWeights()
	newWeights := []float64{
		clip(weights[0]+t.learningRate*angleGrad, t.config.WeightClipMin, t.config.WeightClipMax),
		clip(weights[1]+t.learningRate*angularVelGrad, t.config.WeightClipMin, t.config.WeightClipMax),
		clip(weights[2]+t.learningRate*biasGrad, t.config.WeightClipMin, t.config.WeightClipMax),
	}

	// Update network weights
	t.network.SetWeights(newWeights)

	// Log batch results
	t.logger.Printf("\n[Trainer] Batch %d processed with %d experiences", 
		t.episode/t.config.BatchSize, len(t.batch.Experiences))
	t.logger.Printf("[Trainer] Average reward: %.4f", totalReward/batchSize)
	t.logger.Printf("[Trainer] Weight updates: angle=%.4f, angularVel=%.4f, bias=%.4f",
		angleGrad, angularVelGrad, biasGrad)
	t.logger.Printf("[Trainer] New weights: angle=%.4f, angularVel=%.4f, bias=%.4f",
		newWeights[0], newWeights[1], newWeights[2])

	// Reset batch
	t.batch.Experiences = t.batch.Experiences[:0]
}

// OnEpisodeEnd handles end-of-episode processing
func (t *Trainer) OnEpisodeEnd(episodeTicks int) {
	// Process any remaining experiences in the batch
	t.processBatch()

	// Update learning rate with decay
	t.learningRate = math.Max(
		t.config.MinLearningRate,
		t.learningRate*t.config.LearningRateDecay,
	)

	// Save checkpoint if needed
	if t.episode%t.config.CheckpointInterval == 0 {
		t.saveCheckpoint()
	}

	// Update episode counter
	t.episode++
}

// saveCheckpoint saves the current network state and training metrics
func (t *Trainer) saveCheckpoint() {
	// Create checkpoints directory if it doesn't exist
	checkpointDir := "checkpoints"
	if err := os.MkdirAll(checkpointDir, 0755); err != nil {
		t.logger.Printf("[Trainer] Error creating checkpoint directory: %v", err)
		return
	}

	// Save network weights
	weights := t.network.GetWeights()
	weightsFile := filepath.Join(checkpointDir, fmt.Sprintf("weights_episode_%d.json", t.episode))
	weightsData := struct {
		Episode  int       `json:"episode"`
		Weights  []float64 `json:"weights"`
		DateTime time.Time `json:"datetime"`
	}{
		Episode:  t.episode,
		Weights:  weights,
		DateTime: time.Now(),
	}

	if data, err := json.MarshalIndent(weightsData, "", "  "); err == nil {
		if err := os.WriteFile(weightsFile, data, 0644); err != nil {
			t.logger.Printf("[Trainer] Error saving weights: %v", err)
		}
	}

	// Save metrics
	if len(t.metrics) > 0 {
		metricsFile := filepath.Join(checkpointDir, fmt.Sprintf("metrics_episode_%d.json", t.episode))
		if data, err := json.MarshalIndent(t.metrics, "", "  "); err == nil {
			if err := os.WriteFile(metricsFile, data, 0644); err != nil {
				t.logger.Printf("[Trainer] Error saving metrics: %v", err)
			}
		}
	}

	t.logger.Printf("[Trainer] Checkpoint saved at episode %d", t.episode)
}

// sign returns the sign of a number: 1 for positive, -1 for negative, 0 for zero
func sign(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	if x < 0 {
		return -1.0
	}
	return 0.0
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

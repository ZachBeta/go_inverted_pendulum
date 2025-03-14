package training

import (
	"time"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
)

// Experience represents a single training example
type Experience struct {
	State       env.State
	Action      float64
	Reward      float64
	NextState   env.State
	Done        bool
	TimeStep    uint64
}

// Batch represents a collection of experiences for batch learning
type Batch struct {
	Experiences []Experience
	StartTime   time.Time
	EndTime     time.Time
	EpisodeID   int
}

// Metrics tracks training progress and performance
type Metrics struct {
	EpisodeID          int
	EpisodeTicks       int
	TotalReward        float64
	AverageReward      float64
	MaxAngleDeviation  float64
	MinAngleDeviation  float64
	CartTravel         float64
	WeightUpdates      int
	StartTime          time.Time
	EndTime            time.Time
}

// Config holds training hyperparameters
type Config struct {
	BatchSize           int     // Number of experiences per batch
	BaseLearningRate    float64 // Initial learning rate
	MinLearningRate     float64 // Minimum learning rate
	LearningRateDecay   float64 // Learning rate decay factor
	CheckpointInterval  int     // Episodes between checkpoints
	MaxEpisodes         int     // Maximum number of training episodes
	TargetEpisodeTicks  int     // Target number of ticks per episode
	WeightClipMin       float64 // Minimum weight value
	WeightClipMax       float64 // Maximum weight value
}

// NewDefaultConfig returns a Config with reasonable default values
func NewDefaultConfig() Config {
	return Config{
		BatchSize:           32,
		BaseLearningRate:    0.05,
		MinLearningRate:     0.001,
		LearningRateDecay:   0.995,
		CheckpointInterval:  100,
		MaxEpisodes:         10000,
		TargetEpisodeTicks:  1000,
		WeightClipMin:       -3.0,
		WeightClipMax:       3.0,
	}
}

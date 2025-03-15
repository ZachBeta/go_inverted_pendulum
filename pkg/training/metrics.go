package training

import (
	"fmt"
	"math"
	"time"
)

// MetricsCollector tracks training progress metrics
type MetricsCollector struct {
	EpisodeID       int
	StartTime       time.Time
	MaxAngle        float64
	MinAngle        float64
	TotalReward     float64
	ExperienceCount int
	WeightUpdates   []WeightUpdate
	BatchCount      int // Number of batches processed
}

// WeightUpdate tracks changes in network weights
type WeightUpdate struct {
	Angle      float64
	AngularVel float64
	Bias       float64
	Timestamp  time.Time
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(episodeID int) *MetricsCollector {
	return &MetricsCollector{
		EpisodeID:      episodeID,
		StartTime:      time.Now(),
		MaxAngle:       -math.MaxFloat64,
		MinAngle:       math.MaxFloat64,
		WeightUpdates:  make([]WeightUpdate, 0),
		BatchCount:     0, // Initialize BatchCount to 0
	}
}

// RecordExperience updates metrics with a new experience
func (m *MetricsCollector) RecordExperience(exp Experience) {
	m.ExperienceCount++
	m.TotalReward += exp.Reward
	
	angle := math.Abs(exp.State.AngleRadians)
	if angle > m.MaxAngle {
		m.MaxAngle = angle
	}
	if angle < m.MinAngle {
		m.MinAngle = angle
	}
}

// RecordBatchProcessed increments the batch counter
func (m *MetricsCollector) RecordBatchProcessed() {
	m.BatchCount++
}

// RecordWeightUpdate adds a weight update to the history
func (m *MetricsCollector) RecordWeightUpdate(angle, angularVel, bias float64) {
	m.WeightUpdates = append(m.WeightUpdates, WeightUpdate{
		Angle:      angle,
		AngularVel: angularVel,
		Bias:       bias,
		Timestamp:  time.Now(),
	})
}

// String returns a formatted summary of the metrics
func (m *MetricsCollector) String() string {
	duration := time.Since(m.StartTime)
	avgReward := 0.0
	if m.ExperienceCount > 0 {
		avgReward = m.TotalReward / float64(m.ExperienceCount)
	}

	var lastWeights WeightUpdate
	if len(m.WeightUpdates) > 0 {
		lastWeights = m.WeightUpdates[len(m.WeightUpdates)-1]
	}

	return fmt.Sprintf(`[Training] Episode %d Summary
├── Duration: %.1fs
├── Experiences: %d
├── Batches: %d
├── Max Angle: %.1f°
├── Min Angle: %.1f°
├── Avg Reward: %.3f
└── Final Weights
    ├── Angle: %.3f
    ├── Angular Velocity: %.3f
    └── Bias: %.3f`,
		m.EpisodeID,
		duration.Seconds(),
		m.ExperienceCount,
		m.BatchCount,
		m.MaxAngle * 180 / math.Pi,
		m.MinAngle * 180 / math.Pi,
		avgReward,
		lastWeights.Angle,
		lastWeights.AngularVel,
		lastWeights.Bias,
	)
}

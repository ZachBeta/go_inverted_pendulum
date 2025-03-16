package metrics

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"sync"
	"time"
)

// Logger provides a structured interface for recording neural network performance metrics
type Logger struct {
	db                *DB
	sessionID         string
	episode           int
	step              int
	debug             bool
	stdLogger         *log.Logger
	mu                sync.Mutex
	logFrequency      int  // Log to console every N episodes
	logStepFrequency  int  // Log to console every N steps within an episode
	lastConsoleLog    time.Time
	minLogInterval    time.Duration // Minimum time between console logs
}

// NewLogger creates a new metrics logger with SQLite storage
func NewLogger(dbPath string, debug bool, stdLogger *log.Logger) (*Logger, error) {
	// Use default path if not specified
	if dbPath == "" {
		dbPath = filepath.Join("data", "metrics.db")
	}

	// Create metrics database
	db, err := NewDB(dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics database: %w", err)
	}

	// Generate a unique session ID
	sessionID := GenerateSessionID()

	return &Logger{
		db:               db,
		sessionID:        sessionID,
		episode:          0,
		step:             0,
		debug:            debug,
		stdLogger:        stdLogger,
		logFrequency:     10,  // Default: log to console every 10 episodes
		logStepFrequency: 100, // Default: log to console every 100 steps
		lastConsoleLog:   time.Now(),
		minLogInterval:   2 * time.Second, // Minimum 2 seconds between console logs
	}, nil
}

// SetLogFrequency sets how often to log to the console (in episodes)
func (l *Logger) SetLogFrequency(episodeFreq, stepFreq int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.logFrequency = episodeFreq
	l.logStepFrequency = stepFreq
}

// Close closes the underlying database connection
func (l *Logger) Close() error {
	return l.db.Close()
}

// SetEpisode sets the current episode number
func (l *Logger) SetEpisode(episode int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.episode = episode
	l.step = 0 // Reset step counter for new episode
	
	// Always log episode start to database
	metadata := map[string]interface{}{
		"action": "episode_start",
		"timestamp": time.Now().Format(time.RFC3339),
	}
	metadataJSON, _ := json.Marshal(metadata)
	l.db.RecordMetric(l.sessionID, episode, 0, "system", "episode_start", float64(episode), string(metadataJSON))
	
	// Log episode start to console only at specified frequency
	if l.debug && episode%l.logFrequency == 0 {
		now := time.Now()
		if now.Sub(l.lastConsoleLog) >= l.minLogInterval {
			l.stdLogger.Printf("[Metrics] Starting episode %d", episode)
			l.lastConsoleLog = now
		}
	}
}

// IncrementStep increments the current step counter
func (l *Logger) IncrementStep() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.step++
}

// GetSessionID returns the current session ID
func (l *Logger) GetSessionID() string {
	return l.sessionID
}

// GetCurrentEpisode returns the current episode number
func (l *Logger) GetCurrentEpisode() int {
	return l.episode
}

// GetCurrentStep returns the current step number
func (l *Logger) GetCurrentStep() int {
	return l.step
}

// shouldLogToConsole determines if we should log to the console based on frequency settings
func (l *Logger) shouldLogToConsole() bool {
	if !l.debug {
		return false
	}
	
	// Check if we're at a logging interval for episode or step
	episodeLog := l.episode%l.logFrequency == 0
	stepLog := l.step%l.logStepFrequency == 0
	
	// Also enforce a minimum time between logs
	now := time.Now()
	timeOK := now.Sub(l.lastConsoleLog) >= l.minLogInterval
	
	if (episodeLog || stepLog) && timeOK {
		l.lastConsoleLog = now
		return true
	}
	
	return false
}

// LogWeights records the current network weights
func (l *Logger) LogWeights(angleWeight, angularVelWeight, bias, learningRate float64) error {
	// Always log to database
	err := l.db.RecordWeights(l.sessionID, l.episode, angleWeight, angularVelWeight, bias, learningRate)
	
	// Selectively log to console
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Weights (ep:%d): angle=%.4f, angularVel=%.4f, bias=%.4f, lr=%.4f",
			l.episode, angleWeight, angularVelWeight, bias, learningRate)
	}
	
	return err
}

// LogForwardPass records metrics from a forward pass
func (l *Logger) LogForwardPass(angle, angularVel, force, hidden float64) error {
	// Always log to database
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "input", "angle", angle, ""); err != nil {
		return err
	}
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "input", "angular_vel", angularVel, ""); err != nil {
		return err
	}
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "output", "force", force, ""); err != nil {
		return err
	}
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "hidden", "activation", hidden, ""); err != nil {
		return err
	}
	
	// Selectively log to console
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Forward (ep:%d,step:%d): angle=%.2f, vel=%.2f → force=%.2f",
			l.episode, l.step, angle, angularVel, force)
	}
	
	return nil
}

// LogPrediction records a state value prediction
func (l *Logger) LogPrediction(angle, angularVel, stateValue float64) error {
	// Always log to database
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "prediction", "state_value", stateValue, ""); err != nil {
		return err
	}
	
	// Add additional metadata about the state
	metadata := map[string]float64{
		"angle": angle,
		"angular_vel": angularVel,
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal prediction metadata: %w", err)
	}
	
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "prediction", "state_context", stateValue, string(metadataJSON)); err != nil {
		return err
	}
	
	// Selectively log to console
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Prediction (ep:%d,step:%d): angle=%.2f, vel=%.2f → value=%.2f",
			l.episode, l.step, angle, angularVel, stateValue)
	}
	
	return nil
}

// LogUpdate records a weight update
func (l *Logger) LogUpdate(reward, angleUpdate, angularVelUpdate, biasUpdate float64) error {
	// Always log to database
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "update", "reward", reward, ""); err != nil {
		return err
	}
	
	metadata := map[string]float64{
		"angle_update":      angleUpdate,
		"angular_vel_update": angularVelUpdate,
		"bias_update":        biasUpdate,
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal update metadata: %w", err)
	}
	
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "update", "weight_updates", 
		angleUpdate*angleUpdate+angularVelUpdate*angularVelUpdate+biasUpdate*biasUpdate, 
		string(metadataJSON)); err != nil {
		return err
	}
	
	// Selectively log to console
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Update (ep:%d,step:%d): reward=%.4f, updates=[%.4f, %.4f, %.4f]",
			l.episode, l.step, reward, angleUpdate, angularVelUpdate, biasUpdate)
	}
	
	return nil
}

// LogEpisodeResult records the results of a training episode
func (l *Logger) LogEpisodeResult(totalReward float64, balanceTime int, maxAngle float64, steps int, success bool) error {
	// Always log to database
	if err := l.db.RecordEpisode(l.sessionID, l.episode, totalReward, balanceTime, maxAngle, steps, success); err != nil {
		return err
	}
	
	// Add additional detailed metrics about episode completion
	metadata := map[string]interface{}{
		"success": success,
		"steps": steps,
		"balance_time": balanceTime,
		"max_angle": maxAngle,
		"timestamp": time.Now().Format(time.RFC3339),
	}
	metadataJSON, _ := json.Marshal(metadata)
	l.db.RecordMetric(l.sessionID, l.episode, steps, "system", "episode_complete", totalReward, string(metadataJSON))
	
	// Selectively log to console
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Episode %d result: reward=%.4f, balance_time=%d, max_angle=%.4f, success=%v",
			l.episode, totalReward, balanceTime, maxAngle, success)
	}
	
	return nil
}

// GetSessionSummary returns a summary of the current training session
func (l *Logger) GetSessionSummary() (map[string]interface{}, error) {
	summary, err := l.db.GetSessionSummary(l.sessionID)
	if err != nil {
		return nil, err
	}
	
	// Record summary to database for persistence
	summaryJSON, _ := json.Marshal(summary)
	l.db.RecordMetric(l.sessionID, l.episode, l.step, "system", "session_summary", 
		float64(summary["episode_count"].(int)), string(summaryJSON))
	
	// Only log to console if in debug mode and at appropriate frequency
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Session summary: episodes=%v, success_rate=%.2f%%", 
			summary["episode_count"], 
			summary["success_rate"].(float64)*100)
	}
	
	return summary, nil
}

// GetEpisodeData returns detailed data for a specific episode
func (l *Logger) GetEpisodeData(episode int) (map[string]interface{}, error) {
	data, err := l.db.GetEpisodeData(l.sessionID, episode)
	
	// Record retrieval to database
	if err == nil {
		metadata := map[string]interface{}{
			"action": "retrieve_episode_data",
			"episode": episode,
			"timestamp": time.Now().Format(time.RFC3339),
		}
		metadataJSON, _ := json.Marshal(metadata)
		l.db.RecordMetric(l.sessionID, l.episode, l.step, "system", "data_retrieval", float64(episode), string(metadataJSON))
	}
	
	// Only log to console if in debug mode and at appropriate frequency
	if err == nil && l.shouldLogToConsole() && episode%l.logFrequency == 0 {
		l.stdLogger.Printf("[Metrics] Retrieved data for episode %d", episode)
	}
	
	return data, err
}

// LogNetworkOperation records a network operation such as save/load
func (l *Logger) LogNetworkOperation(operation string, path string, success bool) error {
	metadata := map[string]interface{}{
		"operation": operation,
		"path": path,
		"success": success,
		"timestamp": time.Now().Format(time.RFC3339),
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal operation metadata: %w", err)
	}
	
	value := 1.0
	if !success {
		value = 0.0
	}
	
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "system", "network_operation", value, string(metadataJSON)); err != nil {
		return err
	}
	
	// Selectively log to console
	if l.debug {
		l.stdLogger.Printf("[Metrics] Network %s: path=%s, success=%v", operation, path, success)
	}
	
	return nil
}

// LogTrainingProgress records overall training progress metrics
func (l *Logger) LogTrainingProgress(episodeCount int, successRate float64, avgReward float64, weightChanges map[string]float64) error {
	metadata := map[string]interface{}{
		"episode_count": episodeCount,
		"success_rate": successRate,
		"avg_reward": avgReward,
		"weight_changes": weightChanges,
		"timestamp": time.Now().Format(time.RFC3339),
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal training progress metadata: %w", err)
	}
	
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "system", "training_progress", successRate, string(metadataJSON)); err != nil {
		return err
	}
	
	// Selectively log to console
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Training progress: episodes=%d, success_rate=%.2f%%, avg_reward=%.4f",
			episodeCount, successRate*100, avgReward)
	}
	
	return nil
}

// LogLearningDetail records detailed information about the learning process
func (l *Logger) LogLearningDetail(stateAngle, stateVelocity, predictedValue, actualReward, tdError float64) error {
	// Always log to database
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "learning", "td_error", tdError, ""); err != nil {
		return err
	}
	
	// Add additional metadata about the learning process
	metadata := map[string]interface{}{
		"state_angle": stateAngle,
		"state_velocity": stateVelocity,
		"predicted_value": predictedValue,
		"actual_reward": actualReward,
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal learning metadata: %w", err)
	}
	
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "learning", "state_reward_comparison", actualReward-predictedValue, string(metadataJSON)); err != nil {
		return err
	}
	
	// Selectively log to console
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Learning (ep:%d,step:%d): angle=%.2f, vel=%.2f, prediction=%.2f, reward=%.2f, error=%.2f",
			l.episode, l.step, stateAngle, stateVelocity, predictedValue, actualReward, tdError)
	}
	
	return nil
}

// LogWeightUpdateDetails records detailed information about weight updates
func (l *Logger) LogWeightUpdateDetails(angle, angularVel, force, reward float64, 
	angleUpdate, angularVelUpdate, biasUpdate float64, learningRate float64) error {
	
	// Create detailed metadata
	metadata := map[string]interface{}{
		"angle": angle,
		"angular_vel": angularVel,
		"force": force,
		"reward": reward,
		"learning_rate": learningRate,
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal weight update metadata: %w", err)
	}
	
	// Log individual weight updates with context
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "update", "angle_weight", angleUpdate, string(metadataJSON)); err != nil {
		return err
	}
	
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "update", "angular_vel_weight", angularVelUpdate, string(metadataJSON)); err != nil {
		return err
	}
	
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "update", "bias", biasUpdate, string(metadataJSON)); err != nil {
		return err
	}
	
	// Log total update magnitude
	updateMagnitude := math.Abs(angleUpdate) + math.Abs(angularVelUpdate) + math.Abs(biasUpdate)
	if err := l.db.RecordMetric(l.sessionID, l.episode, l.step, "update", "magnitude", updateMagnitude, ""); err != nil {
		return err
	}
	
	// Selectively log to console
	if l.shouldLogToConsole() {
		l.stdLogger.Printf("[Metrics] Weight Updates (ep:%d,step:%d): angle=%.4f, angularVel=%.4f, bias=%.4f, magnitude=%.4f",
			l.episode, l.step, angleUpdate, angularVelUpdate, biasUpdate, updateMagnitude)
	}
	
	return nil
}

// AnalyzeLearningProgress performs analysis on the learning progress
func (l *Logger) AnalyzeLearningProgress(lastNEpisodes int) (map[string]interface{}, error) {
	return l.db.GetLearningProgress(l.sessionID, lastNEpisodes)
}

// AnalyzePredictionAccuracy analyzes prediction accuracy for a specific episode
func (l *Logger) AnalyzePredictionAccuracy(episode int) (map[string]interface{}, error) {
	return l.db.GetPredictionAccuracy(l.sessionID, episode)
}

// AnalyzeWeightChanges analyzes weight changes for a specific episode
func (l *Logger) AnalyzeWeightChanges(episode int) (map[string]interface{}, error) {
	return l.db.GetWeightChangeAnalysis(l.sessionID, episode)
}

// DetectLearningIssues identifies potential learning problems
func (l *Logger) DetectLearningIssues() (map[string]interface{}, error) {
	return l.db.DetectLearningIssues(l.sessionID)
}

// LogReward records a reward value
func (l *Logger) LogReward(rewardType string, value float64) error {
	return l.db.RecordMetric(l.sessionID, l.episode, l.step, "reward", rewardType, value, "")
}

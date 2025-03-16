package metrics

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3" // SQLite driver
	"math"
)

// DB manages the connection to the SQLite database for performance metrics
type DB struct {
	db     *sql.DB
	mu     sync.Mutex
	dbPath string
}

// NewDB creates a new metrics database connection
func NewDB(dbPath string) (*DB, error) {
	// Ensure directory exists
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	// Open database connection
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Create metrics database
	metricsDB := &DB{
		db:     db,
		dbPath: dbPath,
	}

	// Initialize schema
	if err := metricsDB.initSchema(); err != nil {
		db.Close()
		return nil, err
	}

	return metricsDB, nil
}

// Close closes the database connection
func (m *DB) Close() error {
	return m.db.Close()
}

// initSchema creates the necessary tables if they don't exist
func (m *DB) initSchema() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Create network_metrics table
	_, err := m.db.Exec(`
		CREATE TABLE IF NOT EXISTS network_metrics (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			session_id TEXT,
			episode INTEGER,
			step INTEGER,
			metric_type TEXT,
			metric_name TEXT,
			value REAL,
			metadata TEXT
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create network_metrics table: %w", err)
	}

	// Create network_weights table
	_, err = m.db.Exec(`
		CREATE TABLE IF NOT EXISTS network_weights (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			session_id TEXT,
			episode INTEGER,
			angle_weight REAL,
			angular_vel_weight REAL,
			bias REAL,
			learning_rate REAL
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create network_weights table: %w", err)
	}

	// Create training_episodes table
	_, err = m.db.Exec(`
		CREATE TABLE IF NOT EXISTS training_episodes (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			session_id TEXT,
			episode INTEGER,
			total_reward REAL,
			balance_time INTEGER,
			max_angle REAL,
			steps INTEGER,
			success BOOLEAN
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create training_episodes table: %w", err)
	}

	// Create indices for faster queries
	_, err = m.db.Exec(`
		CREATE INDEX IF NOT EXISTS idx_network_metrics_session_episode ON network_metrics(session_id, episode);
		CREATE INDEX IF NOT EXISTS idx_network_weights_session_episode ON network_weights(session_id, episode);
		CREATE INDEX IF NOT EXISTS idx_training_episodes_session ON training_episodes(session_id);
	`)
	if err != nil {
		return fmt.Errorf("failed to create indices: %w", err)
	}

	return nil
}

// RecordMetric records a single metric value
func (m *DB) RecordMetric(sessionID string, episode, step int, metricType, metricName string, value float64, metadata string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, err := m.db.Exec(`
		INSERT INTO network_metrics (
			session_id, episode, step, metric_type, metric_name, value, metadata
		) VALUES (?, ?, ?, ?, ?, ?, ?)
	`, sessionID, episode, step, metricType, metricName, value, metadata)

	if err != nil {
		return fmt.Errorf("failed to record metric: %w", err)
	}

	return nil
}

// RecordWeights records the current network weights
func (m *DB) RecordWeights(sessionID string, episode int, angleWeight, angularVelWeight, bias, learningRate float64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, err := m.db.Exec(`
		INSERT INTO network_weights (
			session_id, episode, angle_weight, angular_vel_weight, bias, learning_rate
		) VALUES (?, ?, ?, ?, ?, ?)
	`, sessionID, episode, angleWeight, angularVelWeight, bias, learningRate)

	if err != nil {
		return fmt.Errorf("failed to record weights: %w", err)
	}

	return nil
}

// RecordEpisode records training episode results
func (m *DB) RecordEpisode(sessionID string, episode int, totalReward float64, balanceTime int, maxAngle float64, steps int, success bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, err := m.db.Exec(`
		INSERT INTO training_episodes (
			session_id, episode, total_reward, balance_time, max_angle, steps, success
		) VALUES (?, ?, ?, ?, ?, ?, ?)
	`, sessionID, episode, totalReward, balanceTime, maxAngle, steps, success)

	if err != nil {
		return fmt.Errorf("failed to record episode: %w", err)
	}

	return nil
}

// GetSessionSummary returns summary statistics for a training session
func (m *DB) GetSessionSummary(sessionID string) (map[string]interface{}, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result map[string]interface{} = make(map[string]interface{})

	// Get episode count
	var episodeCount int
	err := m.db.QueryRow(`
		SELECT COUNT(*) FROM training_episodes WHERE session_id = ?
	`, sessionID).Scan(&episodeCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get episode count: %w", err)
	}
	result["episode_count"] = episodeCount

	// Get success rate
	var successCount int
	err = m.db.QueryRow(`
		SELECT COUNT(*) FROM training_episodes WHERE session_id = ? AND success = 1
	`, sessionID).Scan(&successCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get success count: %w", err)
	}
	
	if episodeCount > 0 {
		result["success_rate"] = float64(successCount) / float64(episodeCount)
	} else {
		result["success_rate"] = 0.0
	}

	// Get average reward
	var avgReward float64
	err = m.db.QueryRow(`
		SELECT AVG(total_reward) FROM training_episodes WHERE session_id = ?
	`, sessionID).Scan(&avgReward)
	if err != nil {
		return nil, fmt.Errorf("failed to get average reward: %w", err)
	}
	result["avg_reward"] = avgReward

	// Get average balance time
	var avgBalanceTime float64
	err = m.db.QueryRow(`
		SELECT AVG(balance_time) FROM training_episodes WHERE session_id = ?
	`, sessionID).Scan(&avgBalanceTime)
	if err != nil {
		return nil, fmt.Errorf("failed to get average balance time: %w", err)
	}
	result["avg_balance_time"] = avgBalanceTime

	// Get first and last weights to measure learning progress
	var firstEpisode, lastEpisode int
	err = m.db.QueryRow(`
		SELECT MIN(episode), MAX(episode) FROM network_weights WHERE session_id = ?
	`, sessionID).Scan(&firstEpisode, &lastEpisode)
	if err != nil {
		return nil, fmt.Errorf("failed to get episode range: %w", err)
	}

	if firstEpisode != lastEpisode {
		var firstAngleWeight, firstAngularVelWeight, firstBias float64
		var lastAngleWeight, lastAngularVelWeight, lastBias float64

		err = m.db.QueryRow(`
			SELECT angle_weight, angular_vel_weight, bias 
			FROM network_weights 
			WHERE session_id = ? AND episode = ?
		`, sessionID, firstEpisode).Scan(&firstAngleWeight, &firstAngularVelWeight, &firstBias)
		if err != nil {
			return nil, fmt.Errorf("failed to get first weights: %w", err)
		}

		err = m.db.QueryRow(`
			SELECT angle_weight, angular_vel_weight, bias 
			FROM network_weights 
			WHERE session_id = ? AND episode = ?
		`, sessionID, lastEpisode).Scan(&lastAngleWeight, &lastAngularVelWeight, &lastBias)
		if err != nil {
			return nil, fmt.Errorf("failed to get last weights: %w", err)
		}

		result["weight_changes"] = map[string]interface{}{
			"angle_weight":      lastAngleWeight - firstAngleWeight,
			"angular_vel_weight": lastAngularVelWeight - firstAngularVelWeight,
			"bias":              lastBias - firstBias,
		}
	}

	return result, nil
}

// GetEpisodeData returns detailed data for a specific episode
func (m *DB) GetEpisodeData(sessionID string, episode int) (map[string]interface{}, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result map[string]interface{} = make(map[string]interface{})

	// Get episode summary
	var totalReward float64
	var balanceTime int
	var maxAngle float64
	var steps int
	var success bool
	var timestamp string

	err := m.db.QueryRow(`
		SELECT total_reward, balance_time, max_angle, steps, success, timestamp
		FROM training_episodes 
		WHERE session_id = ? AND episode = ?
	`, sessionID, episode).Scan(&totalReward, &balanceTime, &maxAngle, &steps, &success, &timestamp)
	if err != nil {
		return nil, fmt.Errorf("failed to get episode data: %w", err)
	}

	result["total_reward"] = totalReward
	result["balance_time"] = balanceTime
	result["max_angle"] = maxAngle
	result["steps"] = steps
	result["success"] = success
	result["timestamp"] = timestamp

	// Get network weights for this episode
	var angleWeight, angularVelWeight, bias, learningRate float64
	err = m.db.QueryRow(`
		SELECT angle_weight, angular_vel_weight, bias, learning_rate
		FROM network_weights 
		WHERE session_id = ? AND episode = ?
	`, sessionID, episode).Scan(&angleWeight, &angularVelWeight, &bias, &learningRate)
	if err != nil {
		return nil, fmt.Errorf("failed to get episode weights: %w", err)
	}

	result["weights"] = map[string]interface{}{
		"angle_weight":      angleWeight,
		"angular_vel_weight": angularVelWeight,
		"bias":              bias,
		"learning_rate":     learningRate,
	}

	return result, nil
}

// GenerateSessionID creates a unique session ID based on timestamp
func GenerateSessionID() string {
	return fmt.Sprintf("session_%s", time.Now().Format("20060102_150405"))
}

// GetLearningProgress analyzes weight changes and prediction accuracy to detect learning issues
func (m *DB) GetLearningProgress(sessionID string, lastNEpisodes int) (map[string]interface{}, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result map[string]interface{} = make(map[string]interface{})

	// Get the latest episode number
	var latestEpisode int
	err := m.db.QueryRow(`
		SELECT MAX(episode) FROM network_weights WHERE session_id = ?
	`, sessionID).Scan(&latestEpisode)
	if err != nil {
		return nil, fmt.Errorf("failed to get latest episode: %w", err)
	}

	// Calculate start episode based on lastNEpisodes
	startEpisode := 0
	if lastNEpisodes > 0 && latestEpisode >= lastNEpisodes {
		startEpisode = latestEpisode - lastNEpisodes + 1
	}

	// Get weight changes across episodes
	rows, err := m.db.Query(`
		SELECT episode, angle_weight, angular_vel_weight, bias
		FROM network_weights 
		WHERE session_id = ? AND episode >= ?
		ORDER BY episode
	`, sessionID, startEpisode)
	if err != nil {
		return nil, fmt.Errorf("failed to get weight history: %w", err)
	}
	defer rows.Close()

	var weightHistory []map[string]interface{}
	var prevAngleWeight, prevAngularVelWeight, prevBias float64
	var firstRow bool = true

	for rows.Next() {
		var episode int
		var angleWeight, angularVelWeight, bias float64
		if err := rows.Scan(&episode, &angleWeight, &angularVelWeight, &bias); err != nil {
			return nil, fmt.Errorf("failed to scan weight row: %w", err)
		}

		entry := map[string]interface{}{
			"episode":           episode,
			"angle_weight":      angleWeight,
			"angular_vel_weight": angularVelWeight,
			"bias":              bias,
		}

		if !firstRow {
			entry["angle_weight_delta"] = angleWeight - prevAngleWeight
			entry["angular_vel_weight_delta"] = angularVelWeight - prevAngularVelWeight
			entry["bias_delta"] = bias - prevBias
			
			// Calculate magnitude of weight change
			deltaSum := math.Abs(angleWeight - prevAngleWeight) + 
				math.Abs(angularVelWeight - prevAngularVelWeight) + 
				math.Abs(bias - prevBias)
			entry["weight_change_magnitude"] = deltaSum
		}

		weightHistory = append(weightHistory, entry)
		prevAngleWeight, prevAngularVelWeight, prevBias = angleWeight, angularVelWeight, bias
		firstRow = false
	}

	result["weight_history"] = weightHistory

	// Analyze reward trends
	rows, err = m.db.Query(`
		SELECT episode, total_reward, success
		FROM training_episodes 
		WHERE session_id = ? AND episode >= ?
		ORDER BY episode
	`, sessionID, startEpisode)
	if err != nil {
		return nil, fmt.Errorf("failed to get reward history: %w", err)
	}
	defer rows.Close()

	var rewardHistory []map[string]interface{}
	var successCount, totalEpisodes int
	var totalReward float64

	for rows.Next() {
		var episode int
		var reward float64
		var success bool
		if err := rows.Scan(&episode, &reward, &success); err != nil {
			return nil, fmt.Errorf("failed to scan reward row: %w", err)
		}

		rewardHistory = append(rewardHistory, map[string]interface{}{
			"episode": episode,
			"reward":  reward,
			"success": success,
		})

		if success {
			successCount++
		}
		totalReward += reward
		totalEpisodes++
	}

	result["reward_history"] = rewardHistory
	
	if totalEpisodes > 0 {
		result["success_rate"] = float64(successCount) / float64(totalEpisodes)
		result["avg_reward"] = totalReward / float64(totalEpisodes)
	}

	// Detect learning stagnation
	if len(weightHistory) >= 5 {
		// Calculate average weight change magnitude over last 5 episodes
		var totalMagnitude float64
		count := 0
		for i := len(weightHistory) - 5; i < len(weightHistory); i++ {
			if i > 0 && weightHistory[i]["weight_change_magnitude"] != nil {
				totalMagnitude += weightHistory[i]["weight_change_magnitude"].(float64)
				count++
			}
		}
		
		var avgMagnitude float64
		if count > 0 {
			avgMagnitude = totalMagnitude / float64(count)
		}
		
		result["avg_weight_change_magnitude"] = avgMagnitude
		result["learning_stagnated"] = avgMagnitude < 0.001 // Threshold for stagnation
	}

	return result, nil
}

// GetPredictionAccuracy analyzes how well the network's predictions match actual outcomes
func (m *DB) GetPredictionAccuracy(sessionID string, episode int) (map[string]interface{}, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result map[string]interface{} = make(map[string]interface{})

	// Get state value predictions and corresponding rewards
	rows, err := m.db.Query(`
		SELECT m1.step, m1.value as prediction, m2.value as reward
		FROM network_metrics m1
		JOIN network_metrics m2 ON m1.session_id = m2.session_id 
			AND m1.episode = m2.episode 
			AND m1.step = m2.step
		WHERE m1.session_id = ? 
			AND m1.episode = ? 
			AND m1.metric_type = 'prediction' 
			AND m1.metric_name = 'state_value'
			AND m2.metric_type = 'reward' 
			AND m2.metric_name = 'immediate'
		ORDER BY m1.step
	`, sessionID, episode)
	
	if err != nil {
		return nil, fmt.Errorf("failed to get prediction data: %w", err)
	}
	defer rows.Close()

	var predictions []map[string]interface{}
	var totalError, count float64

	for rows.Next() {
		var step int
		var prediction, reward float64
		if err := rows.Scan(&step, &prediction, &reward); err != nil {
			return nil, fmt.Errorf("failed to scan prediction row: %w", err)
		}

		// Calculate prediction error
		error := math.Abs(prediction - reward)
		
		predictions = append(predictions, map[string]interface{}{
			"step":       step,
			"prediction": prediction,
			"reward":     reward,
			"error":      error,
		})

		totalError += error
		count++
	}

	result["predictions"] = predictions
	
	if count > 0 {
		result["avg_prediction_error"] = totalError / count
	}

	return result, nil
}

// GetWeightChangeAnalysis provides detailed analysis of how weights change in response to inputs
func (m *DB) GetWeightChangeAnalysis(sessionID string, episode int) (map[string]interface{}, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result map[string]interface{} = make(map[string]interface{})

	// Get input values and corresponding weight updates
	rows, err := m.db.Query(`
		SELECT 
			m1.step, 
			m1.value as angle, 
			m2.value as angular_vel,
			m3.value as angle_update,
			m4.value as angular_vel_update,
			m5.value as bias_update,
			m6.value as reward
		FROM network_metrics m1
		JOIN network_metrics m2 ON m1.session_id = m2.session_id AND m1.episode = m2.episode AND m1.step = m2.step
		JOIN network_metrics m3 ON m1.session_id = m3.session_id AND m1.episode = m3.episode AND m1.step = m3.step
		JOIN network_metrics m4 ON m1.session_id = m4.session_id AND m1.episode = m4.episode AND m1.step = m4.step
		JOIN network_metrics m5 ON m1.session_id = m5.session_id AND m1.episode = m5.episode AND m1.step = m5.step
		JOIN network_metrics m6 ON m1.session_id = m6.session_id AND m1.episode = m6.episode AND m1.step = m6.step
		WHERE m1.session_id = ? 
			AND m1.episode = ? 
			AND m1.metric_type = 'input' AND m1.metric_name = 'angle'
			AND m2.metric_type = 'input' AND m2.metric_name = 'angular_vel'
			AND m3.metric_type = 'update' AND m3.metric_name = 'angle_weight'
			AND m4.metric_type = 'update' AND m4.metric_name = 'angular_vel_weight'
			AND m5.metric_type = 'update' AND m5.metric_name = 'bias'
			AND m6.metric_type = 'reward' AND m6.metric_name = 'immediate'
		ORDER BY m1.step
	`, sessionID, episode)
	
	if err != nil {
		return nil, fmt.Errorf("failed to get weight update data: %w", err)
	}
	defer rows.Close()

	var updates []map[string]interface{}

	for rows.Next() {
		var step int
		var angle, angularVel, angleUpdate, angularVelUpdate, biasUpdate, reward float64
		if err := rows.Scan(&step, &angle, &angularVel, &angleUpdate, &angularVelUpdate, &biasUpdate, &reward); err != nil {
			return nil, fmt.Errorf("failed to scan update row: %w", err)
		}

		updates = append(updates, map[string]interface{}{
			"step":                step,
			"angle":               angle,
			"angular_vel":         angularVel,
			"angle_weight_update": angleUpdate,
			"angular_vel_weight_update": angularVelUpdate,
			"bias_update":         biasUpdate,
			"reward":              reward,
			"update_magnitude":    math.Abs(angleUpdate) + math.Abs(angularVelUpdate) + math.Abs(biasUpdate),
		})
	}

	result["weight_updates"] = updates

	return result, nil
}

// DetectLearningIssues analyzes metrics to identify potential learning problems
func (m *DB) DetectLearningIssues(sessionID string) (map[string]interface{}, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result map[string]interface{} = make(map[string]interface{})
	var issues []string

	// Check for weight stagnation
	rows, err := m.db.Query(`
		SELECT episode, angle_weight, angular_vel_weight, bias
		FROM network_weights 
		WHERE session_id = ? 
		ORDER BY episode DESC
		LIMIT 10
	`, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to get recent weights: %w", err)
	}
	defer rows.Close()

	var weights []map[string]float64
	for rows.Next() {
		var episode int
		var angleWeight, angularVelWeight, bias float64
		if err := rows.Scan(&episode, &angleWeight, &angularVelWeight, &bias); err != nil {
			return nil, fmt.Errorf("failed to scan weight row: %w", err)
		}

		weights = append(weights, map[string]float64{
			"episode":           float64(episode),
			"angle_weight":      angleWeight,
			"angular_vel_weight": angularVelWeight,
			"bias":              bias,
		})
	}

	// Check for weight stagnation
	if len(weights) >= 5 {
		var totalChange float64
		for i := 0; i < len(weights)-1; i++ {
			totalChange += math.Abs(weights[i]["angle_weight"] - weights[i+1]["angle_weight"])
			totalChange += math.Abs(weights[i]["angular_vel_weight"] - weights[i+1]["angular_vel_weight"])
			totalChange += math.Abs(weights[i]["bias"] - weights[i+1]["bias"])
		}
		
		avgChange := totalChange / float64(len(weights)-1) / 3.0 // 3 weights
		result["avg_weight_change"] = avgChange
		
		if avgChange < 0.0001 {
			issues = append(issues, "Weight stagnation detected - weights barely changing")
		}
	}

	// Check for reward trends
	rows, err = m.db.Query(`
		SELECT episode, total_reward
		FROM training_episodes 
		WHERE session_id = ? 
		ORDER BY episode DESC
		LIMIT 10
	`, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to get recent rewards: %w", err)
	}
	defer rows.Close()

	var rewards []float64
	var episodes []int
	for rows.Next() {
		var episode int
		var reward float64
		if err := rows.Scan(&episode, &reward); err != nil {
			return nil, fmt.Errorf("failed to scan reward row: %w", err)
		}

		rewards = append(rewards, reward)
		episodes = append(episodes, episode)
	}

	// Check for reward improvement
	if len(rewards) >= 5 {
		// Reverse arrays to get chronological order
		for i, j := 0, len(rewards)-1; i < j; i, j = i+1, j-1 {
			rewards[i], rewards[j] = rewards[j], rewards[i]
			episodes[i], episodes[j] = episodes[j], episodes[i]
		}
		
		// Simple linear regression to check trend
		var sumX, sumY, sumXY, sumX2 float64
		n := float64(len(rewards))
		
		for i := 0; i < len(rewards); i++ {
			x := float64(episodes[i])
			y := rewards[i]
			sumX += x
			sumY += y
			sumXY += x * y
			sumX2 += x * x
		}
		
		slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
		result["reward_trend_slope"] = slope
		
		if slope <= 0 {
			issues = append(issues, "No improvement in rewards over recent episodes")
		}
	}

	// Check for extreme weight values
	if len(weights) > 0 {
		latest := weights[0]
		if math.Abs(latest["angle_weight"]) > 50 || 
		   math.Abs(latest["angular_vel_weight"]) > 50 || 
		   math.Abs(latest["bias"]) > 50 {
			issues = append(issues, "Extreme weight values detected - possible exploding gradients")
		}
	}

	result["issues"] = issues
	result["issue_count"] = len(issues)

	return result, nil
}

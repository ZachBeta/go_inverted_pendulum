package metrics

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3" // SQLite driver
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

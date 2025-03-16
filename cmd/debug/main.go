package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/zachbeta/go_inverted_pendulum/pkg/metrics"
)

func main() {
	// Define command-line flags
	dbPathFlag := flag.String("db", "data/metrics.db", "Path to metrics database")
	sessionIDFlag := flag.String("session", "", "Session ID to analyze (default: latest session)")
	episodeFlag := flag.Int("episode", -1, "Episode to analyze (default: latest episode)")
	lastNEpisodesFlag := flag.Int("last", 10, "Number of recent episodes to analyze")
	outputFlag := flag.String("output", "console", "Output format (console, json)")
	analysisTypeFlag := flag.String("type", "all", "Type of analysis (all, learning, weights, predictions, issues)")
	verboseFlag := flag.Bool("verbose", false, "Enable verbose output")
	
	flag.Parse()
	
	// Create logger for console output
	logger := log.New(os.Stdout, "[Debug] ", log.LstdFlags)
	
	// Connect to metrics database
	db, err := metrics.NewDB(*dbPathFlag)
	if err != nil {
		logger.Fatalf("Failed to connect to metrics database: %v", err)
	}
	defer db.Close()
	
	// If no session ID provided, use the latest session
	sessionID := *sessionIDFlag
	if sessionID == "" {
		sessions, err := getSessionList(db)
		if err != nil {
			logger.Fatalf("Failed to get session list: %v", err)
		}
		
		if len(sessions) == 0 {
			logger.Fatalf("No training sessions found in database")
		}
		
		// Use the most recent session
		sessionID = sessions[len(sessions)-1]
		logger.Printf("Using latest session: %s", sessionID)
	}
	
	// Create metrics logger with the specified session
	metricsLogger := &metrics.Logger{
		GetSessionID: func() string { return sessionID },
	}
	
	// Determine episode to analyze
	episode := *episodeFlag
	if episode < 0 {
		// Get the latest episode for this session
		latestEpisode, err := getLatestEpisode(db, sessionID)
		if err != nil {
			logger.Fatalf("Failed to get latest episode: %v", err)
		}
		episode = latestEpisode
		logger.Printf("Using latest episode: %d", episode)
	}
	
	// Perform requested analysis
	var result map[string]interface{}
	
	switch strings.ToLower(*analysisTypeFlag) {
	case "all":
		result = analyzeAll(db, sessionID, episode, *lastNEpisodesFlag, *verboseFlag)
	case "learning":
		learningProgress, err := db.GetLearningProgress(sessionID, *lastNEpisodesFlag)
		if err != nil {
			logger.Fatalf("Failed to analyze learning progress: %v", err)
		}
		result = learningProgress
	case "weights":
		weightChanges, err := db.GetWeightChangeAnalysis(sessionID, episode)
		if err != nil {
			logger.Fatalf("Failed to analyze weight changes: %v", err)
		}
		result = weightChanges
	case "predictions":
		predictionAccuracy, err := db.GetPredictionAccuracy(sessionID, episode)
		if err != nil {
			logger.Fatalf("Failed to analyze prediction accuracy: %v", err)
		}
		result = predictionAccuracy
	case "issues":
		learningIssues, err := db.DetectLearningIssues(sessionID)
		if err != nil {
			logger.Fatalf("Failed to detect learning issues: %v", err)
		}
		result = learningIssues
	default:
		logger.Fatalf("Unknown analysis type: %s", *analysisTypeFlag)
	}
	
	// Output results in requested format
	switch strings.ToLower(*outputFlag) {
	case "console":
		printResults(result, *verboseFlag)
	case "json":
		jsonData, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			logger.Fatalf("Failed to marshal results to JSON: %v", err)
		}
		fmt.Println(string(jsonData))
	default:
		logger.Fatalf("Unknown output format: %s", *outputFlag)
	}
}

// getSessionList returns a list of all session IDs in the database
func getSessionList(db *metrics.DB) ([]string, error) {
	// This is a simplified implementation - in a real application,
	// you would query the database for all unique session IDs
	
	// For now, we'll use a direct SQL query
	rows, err := db.DB().Query(`
		SELECT DISTINCT session_id FROM network_weights
		ORDER BY id
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to query sessions: %w", err)
	}
	defer rows.Close()
	
	var sessions []string
	for rows.Next() {
		var sessionID string
		if err := rows.Scan(&sessionID); err != nil {
			return nil, fmt.Errorf("failed to scan session ID: %w", err)
		}
		sessions = append(sessions, sessionID)
	}
	
	return sessions, nil
}

// getLatestEpisode returns the latest episode number for a session
func getLatestEpisode(db *metrics.DB, sessionID string) (int, error) {
	// Query the database for the maximum episode number
	var latestEpisode int
	err := db.DB().QueryRow(`
		SELECT MAX(episode) FROM network_weights
		WHERE session_id = ?
	`, sessionID).Scan(&latestEpisode)
	
	if err != nil {
		return 0, fmt.Errorf("failed to get latest episode: %w", err)
	}
	
	return latestEpisode, nil
}

// analyzeAll performs all available analyses
func analyzeAll(db *metrics.DB, sessionID string, episode, lastNEpisodes int, verbose bool) map[string]interface{} {
	result := make(map[string]interface{})
	
	// Get session summary
	sessionSummary, err := db.GetSessionSummary(sessionID)
	if err == nil {
		result["session_summary"] = sessionSummary
	} else {
		result["session_summary_error"] = err.Error()
	}
	
	// Get episode data
	episodeData, err := db.GetEpisodeData(sessionID, episode)
	if err == nil {
		result["episode_data"] = episodeData
	} else {
		result["episode_data_error"] = err.Error()
	}
	
	// Get learning progress
	learningProgress, err := db.GetLearningProgress(sessionID, lastNEpisodes)
	if err == nil {
		result["learning_progress"] = learningProgress
	} else {
		result["learning_progress_error"] = err.Error()
	}
	
	// Get prediction accuracy
	predictionAccuracy, err := db.GetPredictionAccuracy(sessionID, episode)
	if err == nil {
		result["prediction_accuracy"] = predictionAccuracy
	} else {
		result["prediction_accuracy_error"] = err.Error()
	}
	
	// Get weight change analysis
	weightChanges, err := db.GetWeightChangeAnalysis(sessionID, episode)
	if err == nil {
		result["weight_changes"] = weightChanges
	} else {
		result["weight_changes_error"] = err.Error()
	}
	
	// Detect learning issues
	learningIssues, err := db.DetectLearningIssues(sessionID)
	if err == nil {
		result["learning_issues"] = learningIssues
	} else {
		result["learning_issues_error"] = err.Error()
	}
	
	return result
}

// printResults prints analysis results to the console
func printResults(results map[string]interface{}, verbose bool) {
	// Print session summary if available
	if summary, ok := results["session_summary"].(map[string]interface{}); ok {
		fmt.Println("\n=== SESSION SUMMARY ===")
		fmt.Printf("Episode Count: %v\n", summary["episode_count"])
		fmt.Printf("Success Rate: %.2f%%\n", summary["success_rate"].(float64)*100)
		fmt.Printf("Average Reward: %.4f\n", summary["avg_reward"])
		
		if changes, ok := summary["weight_changes"].(map[string]interface{}); ok {
			fmt.Println("\nWeight Changes:")
			fmt.Printf("  Angle Weight: %+.4f\n", changes["angle_weight"])
			fmt.Printf("  Angular Velocity Weight: %+.4f\n", changes["angular_vel_weight"])
			fmt.Printf("  Bias: %+.4f\n", changes["bias"])
		}
	}
	
	// Print learning issues if available
	if issues, ok := results["learning_issues"].(map[string]interface{}); ok {
		fmt.Println("\n=== LEARNING ISSUES ===")
		if issueList, ok := issues["issues"].([]interface{}); ok {
			if len(issueList) == 0 {
				fmt.Println("No learning issues detected.")
			} else {
				for i, issue := range issueList {
					fmt.Printf("%d. %s\n", i+1, issue)
				}
			}
		}
		
		if avgChange, ok := issues["avg_weight_change"].(float64); ok {
			fmt.Printf("\nAverage Weight Change: %.6f\n", avgChange)
			if avgChange < 0.0001 {
				fmt.Println("WARNING: Weight changes are very small, learning may be stagnating.")
			}
		}
		
		if slope, ok := issues["reward_trend_slope"].(float64); ok {
			fmt.Printf("Reward Trend Slope: %.6f\n", slope)
			if slope <= 0 {
				fmt.Println("WARNING: Rewards are not improving over time.")
			} else {
				fmt.Println("Rewards are improving over time.")
			}
		}
	}
	
	// Print learning progress if available
	if progress, ok := results["learning_progress"].(map[string]interface{}); ok {
		fmt.Println("\n=== LEARNING PROGRESS ===")
		
		if stagnated, ok := progress["learning_stagnated"].(bool); ok {
			if stagnated {
				fmt.Println("WARNING: Learning appears to be stagnated.")
			} else {
				fmt.Println("Learning is progressing normally.")
			}
		}
		
		if avgMagnitude, ok := progress["avg_weight_change_magnitude"].(float64); ok {
			fmt.Printf("Average Weight Change Magnitude: %.6f\n", avgMagnitude)
		}
		
		if successRate, ok := progress["success_rate"].(float64); ok {
			fmt.Printf("Recent Success Rate: %.2f%%\n", successRate*100)
		}
		
		if avgReward, ok := progress["avg_reward"].(float64); ok {
			fmt.Printf("Recent Average Reward: %.4f\n", avgReward)
		}
		
		// Print weight history if verbose
		if verbose {
			if history, ok := progress["weight_history"].([]interface{}); ok && len(history) > 0 {
				fmt.Println("\nWeight History:")
				for i, entry := range history {
					if i > 0 && i < len(history)-1 && len(history) > 10 {
						// Skip middle entries if there are many
						if i == 1 {
							fmt.Println("  ...")
						}
						continue
					}
					
					entryMap := entry.(map[string]interface{})
					fmt.Printf("  Episode %d: ", int(entryMap["episode"].(float64)))
					fmt.Printf("angle=%.4f, ", entryMap["angle_weight"])
					fmt.Printf("angularVel=%.4f, ", entryMap["angular_vel_weight"])
					fmt.Printf("bias=%.4f", entryMap["bias"])
					
					if i > 0 {
						if mag, ok := entryMap["weight_change_magnitude"].(float64); ok {
							fmt.Printf(" (change: %.6f)", mag)
						}
					}
					fmt.Println()
				}
			}
		}
	}
	
	// Print prediction accuracy if available and verbose
	if verbose {
		if accuracy, ok := results["prediction_accuracy"].(map[string]interface{}); ok {
			fmt.Println("\n=== PREDICTION ACCURACY ===")
			
			if avgError, ok := accuracy["avg_prediction_error"].(float64); ok {
				fmt.Printf("Average Prediction Error: %.4f\n", avgError)
			}
			
			if predictions, ok := accuracy["predictions"].([]interface{}); ok && len(predictions) > 0 {
				fmt.Println("\nPrediction Samples:")
				
				// Print a few samples
				maxSamples := 5
				if len(predictions) < maxSamples {
					maxSamples = len(predictions)
				}
				
				for i := 0; i < maxSamples; i++ {
					pred := predictions[i].(map[string]interface{})
					fmt.Printf("  Step %d: ", int(pred["step"].(float64)))
					fmt.Printf("prediction=%.4f, ", pred["prediction"])
					fmt.Printf("actual=%.4f, ", pred["reward"])
					fmt.Printf("error=%.4f\n", pred["error"])
				}
			}
		}
	}
}

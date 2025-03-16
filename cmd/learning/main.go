package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"
	"encoding/csv"

	"github.com/zachbeta/go_inverted_pendulum/pkg/env"
	"github.com/zachbeta/go_inverted_pendulum/pkg/metrics"
	"github.com/zachbeta/go_inverted_pendulum/pkg/neural"
)

const (
	defaultEpisodes    = 100
	defaultStepsPerEp  = 500
	defaultCheckpoints = 5
	defaultOutputDir   = "./learning_output"
	defaultLearningRate = 0.05
)

var (
	episodes      = flag.Int("episodes", defaultEpisodes, "Number of training episodes to run")
	stepsPerEp    = flag.Int("steps", defaultStepsPerEp, "Number of steps per episode")
	checkpoints   = flag.Int("checkpoints", defaultCheckpoints, "Number of checkpoints to save")
	outputDir     = flag.String("output", defaultOutputDir, "Directory to save checkpoints and metrics")
	verbose       = flag.Bool("verbose", false, "Enable verbose output")
	csvOutput     = flag.Bool("csv", false, "Output metrics in CSV format for visualization")
	adaptiveRate  = flag.Bool("adaptive", true, "Use adaptive learning rate based on success rate")
	initialLR     = flag.Float64("lr", defaultLearningRate, "Initial learning rate")
)

func main() {
	flag.Parse()
	
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}
	
	// Set up logging
	logFile, err := os.Create(filepath.Join(*outputDir, "learning.log"))
	if err != nil {
		log.Fatalf("Failed to create log file: %v", err)
	}
	defer logFile.Close()
	
	var logger *log.Logger
	if *verbose {
		// Log to both file and stdout if verbose
		multiWriter := io.MultiWriter(os.Stdout, logFile)
		logger = log.New(multiWriter, "", log.LstdFlags)
	} else {
		// Log to file only if not verbose
		logger = log.New(logFile, "", log.LstdFlags)
	}
	
	// Set up metrics logger
	metricsDBPath := filepath.Join(*outputDir, "metrics.db")
	metricsLogger, err := metrics.NewLogger(metricsDBPath, *verbose, logger)
	if err != nil {
		logger.Fatalf("Failed to create metrics logger: %v", err)
	}
	defer metricsLogger.Close()
	
	// Create network
	network := neural.NewNetwork()
	network.SetLogger(logger)
	network.SetMetricsLogger(metricsLogger)
	network.SetDebug(false) // Only enable debug in the network during specific tests
	
	// Print header
	fmt.Println("======================================")
	fmt.Println("Neural Network Learning Tool")
	fmt.Println("======================================")
	fmt.Printf("Episodes: %d, Steps per episode: %d, Checkpoints: %d\n", 
		*episodes, *stepsPerEp, *checkpoints)
	fmt.Printf("Output directory: %s\n", *outputDir)
	fmt.Println("======================================")
	
	// Run learning tests
	fmt.Println("\nRunning basic learning tests...")
	runNetworkLearnsToBalance(network, logger)
	
	fmt.Println("\nRunning temporal difference prediction tests...")
	runTemporalDifferencePredictions(network, logger)
	
	fmt.Println("\nRunning checkpoint learning tests...")
	runNetworkImprovesThroughCheckpoints(network, logger, *outputDir, *episodes, *stepsPerEp, *checkpoints)
	
	fmt.Println("\nLearning tests completed successfully")
	fmt.Printf("Results saved to %s\n", *outputDir)
}

// runNetworkLearnsToBalance tests that the network progressively learns to balance the pendulum
func runNetworkLearnsToBalance(network *neural.Network, logger *log.Logger) {
	if *verbose {
		logger.Println("=== Testing Network Learns To Balance ===")
	}
	
	// Reset network to ensure clean state
	network = neural.NewNetwork()
	network.SetLogger(logger)
	network.SetDebug(false)
	
	// Define a sequence of training episodes with progressively better states
	episodes := []struct {
		description string
		state       env.State
		reward      float64
		nextState   env.State
	}{
		{
			description: "Starting with large angle",
			state: env.State{AngleRadians: 0.5, AngularVel: 0.8},
			reward: -0.7,
			nextState: env.State{AngleRadians: 0.4, AngularVel: 0.6},
		},
		{
			description: "Improving to medium angle",
			state: env.State{AngleRadians: 0.3, AngularVel: 0.4},
			reward: -0.3,
			nextState: env.State{AngleRadians: 0.2, AngularVel: 0.2},
		},
		{
			description: "Nearly balanced",
			state: env.State{AngleRadians: 0.1, AngularVel: 0.1},
			reward: 0.8,
			nextState: env.State{AngleRadians: 0.05, AngularVel: 0.05},
		},
	}
	
	// Measure initial prediction
	initialPrediction := network.Predict(episodes[0].state.AngleRadians, episodes[0].state.AngularVel)
	fmt.Printf("  Initial prediction: %.4f\n", initialPrediction)
	
	// Train through episodes
	for i, episode := range episodes {
		fmt.Printf("  Episode %d: %s\n", i+1, episode.description)
		
		// Forward pass to get action
		force := network.Forward(episode.state)
		
		// Update with reward
		network.Update(episode.reward)
		
		if *verbose {
			logger.Printf("  Force: %.4f, Reward: %.4f", force, episode.reward)
		}
	}
	
	// Check final prediction
	finalPrediction := network.Predict(episodes[0].state.AngleRadians, episodes[0].state.AngularVel)
	fmt.Printf("  Final prediction: %.4f (%.2f%% change)\n", 
		finalPrediction, 100*(finalPrediction-initialPrediction)/math.Abs(initialPrediction))
	
	// Verify learning occurred
	if finalPrediction <= initialPrediction {
		fmt.Printf("  Network failed to learn\n")
	} else {
		fmt.Printf("  Network learned successfully\n")
	}
}

// runTemporalDifferencePredictions verifies that the network's TD predictions
// accurately reflect state quality
func runTemporalDifferencePredictions(network *neural.Network, logger *log.Logger) {
	if *verbose {
		logger.Println("=== Testing Temporal Difference Predictions ===")
	}
	
	// Reset network
	network = neural.NewNetwork()
	network.SetLogger(logger)
	network.SetDebug(false)
	
	// Train network with basic scenarios
	scenarios := []struct {
		state  env.State
		reward float64
	}{
		// Good states (balanced)
		{env.State{AngleRadians: 0.01, AngularVel: 0.01}, 0.9},
		{env.State{AngleRadians: -0.01, AngularVel: -0.01}, 0.9},
		
		// Bad states (falling)
		{env.State{AngleRadians: 0.5, AngularVel: 1.0}, -0.8},
		{env.State{AngleRadians: -0.5, AngularVel: -1.0}, -0.8},
	}
	
	// Train network on scenarios
	fmt.Println("  Training network on basic scenarios...")
	for i, s := range scenarios {
		if *verbose {
			logger.Printf("  Scenario %d: angle=%.4f, velocity=%.4f, reward=%.4f", 
				i+1, s.state.AngleRadians, s.state.AngularVel, s.reward)
		}
		network.Forward(s.state)
		network.Update(s.reward)
	}
	
	// Evaluate state values
	tests := []struct {
		description string
		state       env.State
		betterState env.State
		worseState  env.State
	}{
		{
			description: "Balanced state should be valued higher than tilted state",
			state:       env.State{AngleRadians: 0.1, AngularVel: 0.1},
			betterState: env.State{AngleRadians: 0.05, AngularVel: 0.05},
			worseState:  env.State{AngleRadians: 0.2, AngularVel: 0.2},
		},
		{
			description: "Slower velocity should be valued higher when angle is same",
			state:       env.State{AngleRadians: 0.2, AngularVel: 0.3},
			betterState: env.State{AngleRadians: 0.2, AngularVel: 0.1},
			worseState:  env.State{AngleRadians: 0.2, AngularVel: 0.5},
		},
	}
	
	// Check predictions
	fmt.Println("  Evaluating state value predictions:")
	successCount := 0
	totalTests := 0
	
	for i, tt := range tests {
		stateValue := network.Predict(tt.state.AngleRadians, tt.state.AngularVel)
		betterValue := network.Predict(tt.betterState.AngleRadians, tt.betterState.AngularVel)
		worseValue := network.Predict(tt.worseState.AngleRadians, tt.worseState.AngularVel)
		
		fmt.Printf("  Test %d: %s\n", i+1, tt.description)
		
		if *verbose {
			logger.Printf("  Current state value: %.4f", stateValue)
			logger.Printf("  Better state value: %.4f", betterValue)
			logger.Printf("  Worse state value: %.4f", worseValue)
		}
		
		totalTests++
		if betterValue > stateValue {
			successCount++
			fmt.Printf("    Better state valued higher\n")
		} else {
			fmt.Printf("    Better state not valued higher\n")
		}
		
		totalTests++
		if worseValue < stateValue {
			successCount++
			fmt.Printf("    Worse state valued lower\n")
		} else {
			fmt.Printf("    Worse state not valued lower\n")
		}
	}
	
	fmt.Printf("  TD Prediction Success Rate: %d/%d (%.1f%%)\n", 
		successCount, totalTests, 100*float64(successCount)/float64(totalTests))
}

// runNetworkImprovesThroughCheckpoints verifies that network performance improves
// across saved and restored checkpoints
func runNetworkImprovesThroughCheckpoints(network *neural.Network, logger *log.Logger, 
	outputDir string, totalEpisodes, stepsPerEpisode, numCheckpoints int) {
	
	if *verbose {
		logger.Println("=== Testing Network Improves Through Checkpoints ===")
	}
	
	// Create checkpoint directory
	checkpointDir := filepath.Join(outputDir, "checkpoints")
	if err := os.MkdirAll(checkpointDir, 0755); err != nil {
		logger.Fatalf("Failed to create checkpoint directory: %v", err)
	}
	
	// Create CSV file for metrics if enabled
	var csvFile *os.File
	var csvWriter *csv.Writer
	if *csvOutput {
		var err error
		csvFile, err = os.Create(filepath.Join(outputDir, "learning_metrics.csv"))
		if err != nil {
			logger.Fatalf("Failed to create CSV file: %v", err)
		}
		defer csvFile.Close()
		
		csvWriter = csv.NewWriter(csvFile)
		defer csvWriter.Flush()
		
		// Write header
		if err := csvWriter.Write([]string{
			"Episode", "Checkpoint", "AvgReward", "MaxAngle", "SuccessRate", 
			"AngleWeight", "VelocityWeight", "Bias", "LearningRate",
		}); err != nil {
			logger.Fatalf("Failed to write CSV header: %v", err)
		}
	}
	
	// Reset network
	network = neural.NewNetwork()
	network.SetLogger(logger)
	network.SetDebug(false)
	
	// Set initial learning rate
	network.SetLearningRate(*initialLR)
	
	// Define evaluation function
	evaluateNetwork := func(net *neural.Network) (float64, float64, float64) {
		// Run 10 episodes and return average reward, max angle, and success rate
		totalReward := 0.0
		maxAngle := 0.0
		successCount := 0
		const episodes = 10
		const timeSteps = 100
		
		for i := 0; i < episodes; i++ {
			// Create a pendulum simulation
			pendulum := env.NewPendulum(env.NewDefaultConfig(), nil)
			episodeReward := 0.0
			episodeMaxAngle := 0.0
			episodeSuccess := true
			
			for j := 0; j < timeSteps; j++ {
				state := pendulum.GetState()
				
				// Track max angle
				absAngle := math.Abs(state.AngleRadians - math.Pi)
				if absAngle > episodeMaxAngle {
					episodeMaxAngle = absAngle
				}
				
				// Check if pendulum has fallen too far
				if absAngle > math.Pi/4 {
					episodeSuccess = false
				}
				
				force := net.Forward(state)
				newState, err := pendulum.Step(force)
				if err != nil {
					// Skip this step if we hit a constraint
					continue
				}
				
				// Calculate reward (simplified for test)
				// Higher reward for more upright pendulum
				reward := 1.0 - math.Abs(newState.AngleRadians - math.Pi) / math.Pi
				episodeReward += reward
			}
			
			totalReward += episodeReward / float64(timeSteps)
			if episodeMaxAngle > maxAngle {
				maxAngle = episodeMaxAngle
			}
			if episodeSuccess {
				successCount++
			}
		}
		
		avgReward := totalReward / float64(episodes)
		successRate := float64(successCount) / float64(episodes)
		return avgReward, maxAngle, successRate
	}
	
	// Measure initial performance
	var initialReward, initialMaxAngle, initialSuccessRate float64
	initialReward, initialMaxAngle, initialSuccessRate = evaluateNetwork(network)
	fmt.Printf("  Initial performance: Reward=%.4f, MaxAngle=%.4f, SuccessRate=%.1f%%\n", 
		initialReward, initialMaxAngle, initialSuccessRate*100)
	
	// Calculate episodes per checkpoint
	episodesPerCheckpoint := totalEpisodes / numCheckpoints
	
	// Train and save checkpoints
	checkpointPerformances := make([]struct {
		reward      float64
		maxAngle    float64
		successRate float64
	}, numCheckpoints+1)
	
	checkpointPerformances[0] = struct {
		reward      float64
		maxAngle    float64
		successRate float64
	}{
		reward:      initialReward,
		maxAngle:    initialMaxAngle,
		successRate: initialSuccessRate,
	}
	
	// Get initial weights for CSV
	weights := network.GetWeights()
	var angleWeight, velocityWeight, bias float64
	angleWeight, velocityWeight, bias = weights[0], weights[1], weights[2]
	
	// Write initial metrics to CSV
	if *csvOutput {
		if err := csvWriter.Write([]string{
			"0", "0", 
			fmt.Sprintf("%.4f", initialReward),
			fmt.Sprintf("%.4f", initialMaxAngle),
			fmt.Sprintf("%.4f", initialSuccessRate),
			fmt.Sprintf("%.4f", angleWeight),
			fmt.Sprintf("%.4f", velocityWeight),
			fmt.Sprintf("%.4f", bias),
			fmt.Sprintf("%.4f", network.GetLearningRate()),
		}); err != nil {
			logger.Fatalf("Failed to write CSV row: %v", err)
		}
		csvWriter.Flush()
	}
	
	for checkpoint := 1; checkpoint <= numCheckpoints; checkpoint++ {
		fmt.Printf("  Training checkpoint %d/%d...\n", checkpoint, numCheckpoints)
		startTime := time.Now()
		
		// Train for this checkpoint phase
		episodeSuccesses := 0
		
		for i := 0; i < episodesPerCheckpoint; i++ {
			// Set episode number for metrics
			episodeNum := (checkpoint-1)*episodesPerCheckpoint + i + 1
			network.SetEpisode(episodeNum)
			
			// Generate experience and train
			pendulum := env.NewPendulum(env.NewDefaultConfig(), nil)
			episodeReward := 0.0
			episodeMaxAngle := 0.0
			episodeSuccess := true
			
			for j := 0; j < stepsPerEpisode; j++ {
				network.IncrementStep()
				
				// Get current state
				state := pendulum.GetState()
				
				// Track max angle
				absAngle := math.Abs(state.AngleRadians - math.Pi)
				if absAngle > episodeMaxAngle {
					episodeMaxAngle = absAngle
				}
				
				// Check if pendulum has fallen too far
				if absAngle > math.Pi/4 {
					episodeSuccess = false
				}
				
				// Get action from network
				force := network.Forward(state)
				
				// Apply action to environment
				newState, err := pendulum.Step(force)
				if err != nil {
					// Skip this step if we hit a constraint
					continue
				}
				
				// Calculate reward
				reward := 1.0 - math.Abs(newState.AngleRadians - math.Pi) / math.Pi
				episodeReward += reward
				
				// Update network with reward
				network.Update(reward)
				
				// TD learning update
				currentValue := network.Predict(state.AngleRadians, state.AngularVel)
				nextValue := network.Predict(newState.AngleRadians, newState.AngularVel)
				
				// We're not using these values directly, but they trigger the TD learning internally
				_ = currentValue
				_ = nextValue
				
				if *verbose && j%100 == 0 {
					logger.Printf("  Episode %d, Step %d: angle=%.4f, reward=%.4f, value=%.4f", 
						i+1, j+1, state.AngleRadians, reward, currentValue)
				}
			}
			
			// Track episode success
			if episodeSuccess {
				episodeSuccesses++
			}
			
			// Adaptive learning rate if enabled
			if *adaptiveRate && i > 0 && i%10 == 0 {
				successRate := float64(episodeSuccesses) / float64(i+1)
				
				// Adjust learning rate based on success rate
				currentLR := network.GetLearningRate()
				var newLR float64
				
				if successRate < 0.3 {
					// Poor performance, reduce learning rate
					newLR = currentLR * 0.9
				} else if successRate > 0.7 {
					// Good performance, increase learning rate slightly
					newLR = currentLR * 1.05
				} else {
					// Moderate performance, keep learning rate stable
					newLR = currentLR
				}
				
				// Ensure learning rate stays within reasonable bounds
				if newLR < 0.001 {
					newLR = 0.001
				} else if newLR > 0.2 {
					newLR = 0.2
				}
				
				if newLR != currentLR {
					network.SetLearningRate(newLR)
					if *verbose {
						logger.Printf("  Adjusted learning rate: %.4f -> %.4f (success rate: %.2f)", 
							currentLR, newLR, successRate)
					}
				}
			}
			
			// Get current weights for CSV
			weights := network.GetWeights()
			angleWeight, velocityWeight, bias = weights[0], weights[1], weights[2]
			
			// Write episode metrics to CSV
			if *csvOutput && (i%5 == 0 || i == episodesPerCheckpoint-1) {
				avgReward := episodeReward / float64(stepsPerEpisode)
				successRate := 0.0
				if episodeSuccess {
					successRate = 1.0
				}
				
				if err := csvWriter.Write([]string{
					fmt.Sprintf("%d", episodeNum),
					fmt.Sprintf("%d", checkpoint),
					fmt.Sprintf("%.4f", avgReward),
					fmt.Sprintf("%.4f", episodeMaxAngle),
					fmt.Sprintf("%.4f", successRate),
					fmt.Sprintf("%.4f", angleWeight),
					fmt.Sprintf("%.4f", velocityWeight),
					fmt.Sprintf("%.4f", bias),
					fmt.Sprintf("%.4f", network.GetLearningRate()),
				}); err != nil {
					logger.Fatalf("Failed to write CSV row: %v", err)
				}
				csvWriter.Flush()
			}
			
			// Log progress every 10 episodes or for the last episode
			if i%10 == 0 || i == episodesPerCheckpoint-1 {
				fmt.Printf("    Episode %d/%d completed\r", i+1, episodesPerCheckpoint)
			}
		}
		fmt.Println() // End the progress line
		
		// Calculate checkpoint success rate
		checkpointSuccessRate := float64(episodeSuccesses) / float64(episodesPerCheckpoint)
		
		// Save checkpoint
		checkpointPath := filepath.Join(checkpointDir, fmt.Sprintf("checkpoint_%d.json", checkpoint))
		if err := network.SaveToFile(checkpointPath); err != nil {
			logger.Fatalf("Failed to save checkpoint: %v", err)
		}
		
		// Evaluate performance
		var reward, maxAngle, successRate float64
		reward, maxAngle, successRate = evaluateNetwork(network)
		checkpointPerformances[checkpoint] = struct {
			reward      float64
			maxAngle    float64
			successRate float64
		}{
			reward:      reward,
			maxAngle:    maxAngle,
			successRate: successRate,
		}
		
		duration := time.Since(startTime)
		fmt.Printf("  Checkpoint %d complete in %v\n", checkpoint, duration)
		fmt.Printf("  Performance: Reward=%.4f, MaxAngle=%.4f, SuccessRate=%.1f%%\n", 
			reward, maxAngle, successRate*100)
		fmt.Printf("  Training success rate: %.1f%%\n", checkpointSuccessRate*100)
	}
	
	// Print summary
	fmt.Println("\n  Training Summary:")
	fmt.Printf("  Initial: Reward=%.4f, MaxAngle=%.4f, SuccessRate=%.1f%%\n", 
		checkpointPerformances[0].reward, 
		checkpointPerformances[0].maxAngle,
		checkpointPerformances[0].successRate*100)
	
	var improved bool
	improved = true
	for i := 1; i <= numCheckpoints; i++ {
		rewardChange := 100 * (checkpointPerformances[i].reward - checkpointPerformances[0].reward) / 
			math.Abs(checkpointPerformances[0].reward)
		angleChange := 100 * (checkpointPerformances[0].maxAngle - checkpointPerformances[i].maxAngle) / 
			checkpointPerformances[0].maxAngle
		successChange := 100 * (checkpointPerformances[i].successRate - checkpointPerformances[0].successRate)
		
		fmt.Printf("  Checkpoint %d: Reward=%.4f (%+.1f%%), MaxAngle=%.4f (%+.1f%%), SuccessRate=%.1f%% (%+.1f%%)\n", 
			i, 
			checkpointPerformances[i].reward, rewardChange,
			checkpointPerformances[i].maxAngle, angleChange,
			checkpointPerformances[i].successRate*100, successChange)
		
		if i > 1 && checkpointPerformances[i].reward <= checkpointPerformances[i-1].reward {
			improved = false
		}
	}
	
	// Get final weights
	weights = network.GetWeights()
	angleWeight, velocityWeight, bias = weights[0], weights[1], weights[2]
	fmt.Printf("\n  Final Network Configuration:\n")
	fmt.Printf("  Angle Weight: %.4f\n", angleWeight)
	fmt.Printf("  Velocity Weight: %.4f\n", velocityWeight)
	fmt.Printf("  Bias: %.4f\n", bias)
	fmt.Printf("  Learning Rate: %.4f\n", network.GetLearningRate())
	
	// Verify overall improvement
	var finalPerf, initialPerf struct {
		reward      float64
		maxAngle    float64
		successRate float64
	}
	finalPerf = checkpointPerformances[numCheckpoints]
	initialPerf = checkpointPerformances[0]
	
	if finalPerf.reward <= initialPerf.reward {
		fmt.Printf("\n  Network did not improve overall\n")
	} else if !improved {
		fmt.Printf("\n  Network improved overall but not consistently between checkpoints\n")
	} else {
		fmt.Printf("\n  Network improved consistently across all checkpoints\n")
	}
	
	// Output CSV file location if used
	if *csvOutput {
		fmt.Printf("\n  CSV metrics saved to: %s\n", filepath.Join(outputDir, "learning_metrics.csv"))
		fmt.Printf("  You can visualize these metrics using any plotting tool or spreadsheet software\n")
	}
}

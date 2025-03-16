# Neural Network Learning Tool

This command-line tool runs learning tests for the neural network implementation, providing detailed output about the network's learning progress.

## Usage

```bash
go run cmd/learning/main.go [options]
```

### Options

- `-episodes int`: Number of training episodes to run (default: 100)
- `-steps int`: Number of steps per episode (default: 500)
- `-checkpoints int`: Number of checkpoints to save (default: 5)
- `-output string`: Directory to save checkpoints and metrics (default: "./learning_output")
- `-verbose`: Enable verbose output (default: false)

## What it Tests

1. **Network Learns to Balance**: Tests that the network progressively learns to balance the pendulum by training on a sequence of progressively better states.

2. **Temporal Difference Predictions**: Verifies that the network's TD predictions accurately reflect state quality, with better states receiving higher value predictions.

3. **Network Improves Through Checkpoints**: Verifies that network performance improves across saved and restored checkpoints, with each checkpoint showing better performance than the previous one.

## Output

The tool generates the following outputs:

- Detailed logs in `learning.log`
- Metrics database in `metrics.db`
- Checkpoint files in the `checkpoints` directory

## Examples

Run with default settings:
```bash
go run cmd/learning/main.go
```

Run with custom settings:
```bash
go run cmd/learning/main.go -episodes 200 -steps 1000 -checkpoints 10 -output "./custom_output" -verbose
```

## Visualizing Results

The metrics database can be used to generate visualizations of the network's learning progress. Use the metrics package to query the database and generate plots.

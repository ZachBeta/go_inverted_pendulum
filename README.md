# Go Inverted Pendulum Neural Network Implementation

A Go-based framework implementing NEAT (NeuroEvolution of Augmenting Topologies) for the inverted pendulum problem, building directly on Pezzza's original approach with a focus on evolvable architectures and performance.

## Overview
This project implements neural network control in Go, focusing on:
- NEAT-based evolvable network architecture
- DAG-based network representation
- Progressive topology evolution
- Temporal difference learning
- Learned action space (-5N to +5N force control)
- Real-time performance optimization
- Environment simulation with accurate physics
- Species diversity management

## Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd go_inverted_pendulum

# Create local package directory (used for dependency management)
mkdir -p .packages
```

### 2. Install Dependencies
```bash
# Install Ebiten using local package management
# Run this from the project root directory
GOPATH="$(pwd)/.packages" go get github.com/hajimehoshi/ebiten/v2@v2.8.6

# Note: The .packages directory is gitignored and used for local dependency management
# For all package installations, use: GOPATH=/path/to/project/.packages go get <package>
```

### 3. Run Demo
```bash
# Run the window demo
go run cmd/window/main.go
```

### 4. Run Tests
```bash
# Run all tests
go test ./...
```

## Project Structure
```
.
├── .packages/     # Local package management directory (gitignored)
├── cmd/           # Command-line applications
│   └── window/   # Window demo application (800x600)
├── internal/      # Private application code
├── pkg/          # Public library code
│   ├── agent/    # Neural network implementations
│   ├── env/      # Environment definitions
│   ├── policy/   # Learning policy implementations
│   └── reward/   # Reward system (in progress)
├── test/         # Additional test files
├── docs/         # Documentation
│   ├── ARCHITECTURE.md  # Architecture decisions and references
│   └── PROGRESS.md     # Project status and roadmap
└── examples/     # Example implementations
```

## Controls
- Left Arrow: Apply -5N force
- Right Arrow: Apply +5N force
- No key: Zero force

## Development
Please read our [RULES.md](RULES.md) for detailed development guidelines and requirements.

For current project status and next steps, see [PROGRESS.md](docs/PROGRESS.md).

## Requirements
- Go 1.21 or higher
- Dependencies managed via local `.packages` directory (see Quick Start)
- [Ebiten](https://github.com/hajimehoshi/ebiten) v2.8.6 (for visualization)

## Implementation Details
This project extends Pezzza's original neural network implementation with insights from his video series. Key features:

### Neural Network Architecture
- NEAT-based evolvable topology
- Starting from minimal structure
- Growing complexity through evolution
- DAG-based representation for efficient processing
- Innovation number tracking
- Species management for diversity

### Training Methodology
- Progressive learning with topology evolution
- Clear success metrics for evaluation
- Temporal difference learning
- Species-based fitness sharing
- Adaptive mutation rates
- Protection of innovation

### Performance Optimization
- Efficient network processing
- Species-parallel evaluation
- Adaptive learning rates
- Network state persistence
- Comprehensive checkpoint system
- Topology-aware processing

### Evolution Features
- Dynamic connection creation
- Strategic node addition
- Weight modification strategies
- Innovation history tracking
- Population management
- Species diversity maintenance

## License
See [LICENSE](LICENSE) file for details.
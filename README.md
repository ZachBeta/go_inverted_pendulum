# Go Inverted Pendulum Classical Control Framework

A Go-based framework implementing classical control theory for the inverted pendulum problem, with a focus on deterministic behavior, performance, and maintainability.

## Overview
This project implements classical control principles in Go, focusing on:
- Immutable state management with DAG-based representation
- Deterministic action space (-5N to +5N force control)
- Real-time performance optimization
- Environment simulation with accurate physics

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
│   ├── agent/    # Agent implementations
│   ├── env/      # Environment definitions
│   ├── policy/   # Policy implementations
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
This project builds on Pezzza's original neural network implementation while taking a classical control approach. Key features:
- Deterministic control vs learned behaviors
- Immutable state patterns for thread safety
- Real-time performance optimization
- Comprehensive testing framework
- Progressive complexity approach

For implementation insights and video references, see our [architecture documentation](docs/ARCHITECTURE.md).

## License
See [LICENSE](LICENSE) file for details.
# Development Rules

## Core Requirements
- Go 1.21+
- Strong testing coverage
- Performance optimization focus
- Clear documentation standards

## Neural Network Architecture
1. Core Components:
   - Three-node design (input, hidden, output)
   - DAG-based representation
   - Weight initialization
   - Topological processing order

2. Training Pipeline:
   - Progressive learning approach
   - Temporal difference (TD) learning
   - Momentum-based backpropagation
   - Adaptive learning rate
   - Checkpoint system for persistence

3. Performance Design:
   - Matrix operations for layered networks
   - Goroutine parallelization
   - Batch processing capability
   - Training optimization

## Project Structure
```
.
├── cmd/          # CLI applications
├── internal/     # Private code
├── pkg/         
│   ├── agent/    # Neural network implementation
│   ├── env/      # Environment simulation
│   ├── policy/   # Learning policies
│   └── reward/   # Reward calculation
├── test/
└── examples/
```

## Visualization (Ebiten v2.8.6)
- Window: 800x600 pixels
- Scale: 100 pixels per meter
- Components:
  - Track: White line at 70% height
  - Cart: Blue 50x30px rectangle
  - Pendulum: Red line with 10px radius bob
- Debug overlay with network state
- Real-time training metrics

## Angle Conventions
- Normalized to [0, 2π]
- Measured clockwise from vertical
- 0 = upward
- π = downward
- π/2 = rightward
- 3π/2 = leftward

## Package Management
- Local `.packages` directory
- Install packages with:
  ```bash
  GOPATH=/path/to/project/.packages go get <package>
  ```

## Documentation Standards
1. Core Documentation:
   - README.md: Project overview
   - ARCHITECTURE.md: Design decisions
   - PROGRESS.md: Implementation status
   - API documentation with examples

2. Code Quality:
   - Use `gofmt`, `goimports`, `govet`
   - Use `golint`, `golangci-lint`
   - Maintain test coverage

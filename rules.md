# Development Rules and Guidelines

## Core Development Tools
* use gofmt and similar tools where appropriate
   * goimports
   * govet
   * golint
   * golangci-lint
* give visibility into the neural network
* add debugging info display on renderer
* log debugging information that's displayed on screen to a file as well

## Core Principles
1. **Learning Strategy**
   - Progressive learning approach
   - Evolutionary architecture design
   - Continuous adaptation and improvement
   - Learning rate optimization
   - Batch processing implementation

2. **Code Organization**
   - Follow standard Go project layout
   - Use interfaces for flexibility and testing
   - Maintain clear separation between components
   - Document architectural decisions in ADRs

## Physics Implementation

### Angle Conventions
- All angles normalized to [0, 2π]
- Measured clockwise from upward vertical
- 0 = upward pointing
- π = downward pointing
- π/2 = rightward pointing
- 3π/2 = leftward pointing

### Simulation Requirements
- Training data generation
- Full nonlinear equations implementation
- Proper collision detection (track bounds)
- Training episode replay capability
- Learning progress logging

## Testing Requirements
1. **Integration Testing**
   - Network training progression
   - Component interaction tests
   - Full system validation
   - Training episode replay

2. **Performance Testing**
   - Training speed benchmarks
   - Memory allocation profiling
   - Real-time performance validation
   - Learning efficiency metrics

## Performance Guidelines
1. **Memory Management**
   - Efficient weight matrix operations
   - Implement batch processing
   - Minimize allocations in training loops
   - Regular memory usage monitoring

2. **Training Optimization**
   - Efficient backpropagation
   - Learning rate adaptation
   - Batch size optimization
   - Training checkpoints

## Progress Tracking
1. **Progress Tracking**
   - Current status in TODO.md
   - Completed components
   - In-progress features
   - Next steps and priorities

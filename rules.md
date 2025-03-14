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
1. **Neural Network Architecture**
   - Three-node network design (input/hidden/output)
   - Use DAG-based network representation
   - Implement weight initialization and updates
   - Track network state and learning progress
   - Maintain training history

2. **Action Space**
   - Force application interface (-5N to +5N)
   - Neural network action generation
   - Implement force clamping and validation
   - Clear state-action-reward pipeline
   - Each action must return both a new state and a reward signal

3. **Learning Strategy**
   - Progressive learning approach
   - Evolutionary architecture design
   - Continuous adaptation and improvement
   - Learning rate optimization
   - Batch processing implementation

4. **Code Organization**
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
1. **Neural Network Testing**
   - Forward propagation validation
   - Backpropagation verification
   - Weight update correctness
   - Learning rate optimization

2. **Integration Testing**
   - Network training progression
   - Component interaction tests
   - Full system validation
   - Training episode replay

3. **Performance Testing**
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

## Documentation Requirements
1. **Code Documentation**
   - Clear API documentation with examples
   - Network architecture details
   - Performance characteristics
   - Core types documentation

2. **Architecture Documentation**
   - ADRs in docs/ARCHITECTURE.md
   - Network design decisions
   - Training methodology
   - Reference implementation comparisons

3. **Progress Tracking**
   - Current status in docs/PROGRESS.md
   - Completed components
   - In-progress features
   - Next steps and priorities

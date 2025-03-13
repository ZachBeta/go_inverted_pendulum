# Development Rules and Guidelines

## Core Development Tools
* use gofmt
* use goimports
* use govet
* use golint
* use golangci-lint

## Core Principles
1. **State Management**
   - Implement immutable state patterns for thread safety
   - Use DAG-based system representation (input/hidden/output nodes)
   - All state changes must be explicit and trackable
   - Implement comprehensive state logging
   - Thread-safe state transitions

2. **Action Space**
   - Force application interface (-5N to +5N)
   - All actions must be deterministic
   - Implement force clamping and validation
   - Clear state-action-reward pipeline
   - Each action must return both a new state and a reward signal

3. **Control Strategy**
   - Classical control approach (vs original neural network implementation)
   - Progressive complexity: start with minimal parameters
   - Clear stage separation for validation
   - Deterministic behavior over learned responses

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
- Configurable random seed support
- Full nonlinear equations implementation
- Proper collision detection (track bounds)
- Deterministic replay capability
- State transition logging

## Testing Requirements
1. **Unit Testing**
   - Core component unit tests
   - State transition validation
   - Action space verification
   - Reward calculation tests

2. **Integration Testing**
   - Component interaction tests
   - Full system validation
   - Deterministic replay tests
   - Random seed testing

3. **Performance Testing**
   - Critical path benchmarks
   - Memory allocation profiling
   - Real-time performance validation
   - Stage-by-stage validation

## Performance Guidelines
1. **Memory Management**
   - Use immutable patterns for thread safety
   - Implement object pooling for frequent allocations
   - Minimize allocations in hot paths
   - Regular memory usage monitoring

2. **Concurrency**
   - Responsible goroutine management
   - Thread-safe state transitions
   - Proper error handling
   - Document thread safety guarantees

## Documentation Requirements
1. **Code Documentation**
   - Clear API documentation with examples
   - Usage constraints and limitations
   - Performance characteristics
   - Core types documentation

2. **Architecture Documentation**
   - ADRs in docs/ARCHITECTURE.md
   - Performance impact analysis
   - Implementation trade-offs
   - Reference implementation comparisons

3. **Progress Tracking**
   - Current status in docs/PROGRESS.md
   - Completed components
   - In-progress features
   - Next steps and priorities

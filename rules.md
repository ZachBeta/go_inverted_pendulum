# Development Rules and Guidelines

## Golang
* use gofmt
* use goimports
* use govet
* use golint
* use golangci-lint

## simulation
* use random seed that we can configure

## Core Principles
1. **State Management**
   - All state changes must be explicit and trackable
   - Use immutable state patterns where possible
   - Implement clear state observation mechanisms

2. **Action Space**
   - Define clear interfaces for all possible actions
   - Actions must be deterministic
   - Each action must return both a new state and a reward signal

3. **Reward System**
   - All rewards must be quantifiable
   - Implement both immediate and delayed reward mechanisms
   - Keep reward calculations consistent and documented

4. **Code Organization**
   - Follow standard Go project layout
   - Use interfaces for flexibility and testing
   - Maintain clear separation between environment, agent, and policy logic

## Angle Conventions

- All angles in the system are normalized to the range [0, 2π]
- The pendulum's angle is measured clockwise from the upward vertical position
- 0 represents the pendulum pointing upward
- π represents the pendulum pointing downward
- π/2 represents the pendulum pointing rightward
- 3π/2 represents the pendulum pointing leftward

## Testing Requirements
1. Each component must have:
   - Unit tests for core logic
   - Integration tests for state transitions
   - Performance benchmarks for critical paths

2. Simulation requirements:
   - Must support deterministic replay
   - Must log all state transitions
   - Must support different random seeds

## Performance Guidelines
1. **Memory Management**
   - Minimize allocations in hot paths
   - Use object pooling for frequently created/destroyed objects
   - Profile memory usage regularly

2. **Concurrency**
   - Use goroutines responsibly
   - Implement proper error handling for concurrent operations
   - Document thread safety guarantees

## Documentation
1. All public APIs must have:
   - Clear documentation with examples
   - Usage constraints and limitations
   - Performance characteristics

2. Architecture decisions must be documented in:
   - ADR (Architecture Decision Records)
   - Performance impact analysis
   - Trade-off considerations

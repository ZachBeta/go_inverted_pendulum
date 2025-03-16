# Neural Network Design Document

## Core Types

### Node
```go
type Node struct {
    ID          int
    Bias        float64
    Value       float64
    Activation  func(float64) float64
    Incoming    []*Connection
    Outgoing    []*Connection
}
```

### Connection
```go
type Connection struct {
    From     *Node
    To       *Node
    Weight   float64
    Enabled  bool
    Innovation int  // For NEAT tracking
}
```

### Network
```go
type Network struct {
    Nodes       []*Node
    Connections []*Connection
    InputNodes  []*Node
    OutputNodes []*Node
    SortedNodes []*Node  // Cached topological sort
}
```

## Core Operations

### Node Processing
1. Reset node values
2. Process nodes in topological order:
   ```go
   func (n *Node) Process() {
       sum := n.Bias
       for _, conn := range n.Incoming {
           if conn.Enabled {
               sum += conn.From.Value * conn.Weight
           }
       }
       n.Value = n.Activation(sum)
   }
   ```

### Network Evolution
1. **New Connection**
   - Add connection between unconnected nodes
   - Ensure DAG property is maintained
   - Assign new innovation number

2. **Split Connection**
   - Disable existing connection
   - Add new node
   - Add two new connections
   - Preserve network behavior initially

3. **Weight Modification**
   - Small random adjustments
   - Momentum-based changes
   - Learning rate adaptation

## Training Pipeline

### Population Management
```go
type Population struct {
    Networks    []*Network
    Generation  int
    TopPercent  float64  // e.g., 0.3 for top 30%
}
```

### Selection Process
1. Evaluate all networks
2. Sort by fitness score
3. Select top performers
4. Random selection weighted by score

### Evolution Steps
1. Create offspring through mutation
2. Maintain population size
3. Track innovation numbers
4. Update generation counter

## Performance Optimization

### Parallel Processing
1. Goroutines for population evaluation
2. Batch processing capabilities
3. Concurrent mutation operations

### Matrix Operations
Future enhancement for layered networks:
```go
type LayeredNetwork struct {
    Layers      [][]Node
    Weights     []Matrix
    BiasVectors []Vector
}
```

## Persistence

### State Management
```go
type NetworkState struct {
    Nodes       []NodeState
    Connections []ConnectionState
    Innovation  int
    Generation  int
}
```

### Checkpoint System
- Regular state saves
- Progress tracking
- Training resumption
- Performance metrics

## Implementation Phases

1. **Core Network (Phase 1)**
   - Node and Connection implementations
   - Basic network operations
   - Topological processing

2. **Evolution (Phase 2)**
   - Mutation operations
   - Population management
   - Selection process

3. **Training (Phase 3)**
   - Fitness evaluation
   - Progressive difficulty
   - Learning rate adaptation

4. **Optimization (Phase 4)**
   - Parallel processing
   - Matrix operations
   - Performance profiling

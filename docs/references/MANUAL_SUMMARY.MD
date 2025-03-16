# Neural Network Implementation Reference

## Core Architecture
- Three-node neural network using NEAT principles
- Directed Acyclic Graph (DAG) representation
- Topological sorting for efficient processing
- Reference: [Pendulum-NEAT Implementation](https://github.com/johnBuffer/Pendulum-NEAT)

## Node Processing Algorithm
1. Input processing:
   - Add bias value
   - Apply activation function
2. Connection handling:
   - Multiply by connection weight
   - Add to destination neuron sum
3. Processing order:
   - Follow topological sorting
   - Enables parallel processing with goroutines

## Training Strategy
1. Evaluation Phase
2. Selection Phase:
   - Top 30% selection
   - Score-based random selection
3. Mutation Phase:
   - New connections
   - Connection splitting
   - Weight modifications

## Performance Optimization
- Layered network support
- Matrix multiplication optimization
- Goroutine parallelization
- Progressive learning approach

## References
- NEAT paper (to be added)
- Wikipedia: Topological Sorting
- Reinforcement Learning principles
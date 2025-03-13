# Reference Materials

## Video References

### Neural Networks Learning to Play Inverted Pendulum
- **Source**: [YouTube Video](https://www.youtube.com/watch?v=EvV5Qtp_fYg)
- **Author**: Pezzza
- **Description**: A demonstration and explanation of using neural networks to solve the inverted pendulum control problem. The video showcases the learning process, challenges, and eventual success of neural networks in mastering pendulum balancing.

The video transcript is available in [inverted_pendulum_video.en.srt](./inverted_pendulum_video.en.srt).

#### Key Points from Video:
1. Machine learning approach to solving the inverted pendulum problem
2. Neural network architecture considerations
3. Training process and iterations
4. Challenges in balancing exploration vs exploitation
5. Final results and performance analysis

#### Relevance to Our Project:
This video by Pezzza serves as the inspiration for our project's visualization and physics simulation. While Pezzza's implementation uses neural networks for control, our project takes a classical control theory approach to solve the same problem. This allows us to:

1. Compare and contrast different solution methodologies
2. Leverage the same physics simulation and visualization framework
3. Benchmark our classical control approach against neural network performance
4. Learn from the challenges and solutions presented in the neural network implementation

The key differences in our approach:
- Using deterministic control algorithms instead of learned behaviors
- Implementing state management with immutable patterns
- Focusing on real-time performance optimization
- Providing a framework for experimenting with different control strategies

#### Implementation Insights from Pezzza's Work:
1. **Architecture Design**
   - Uses directed acyclic graphs (DAGs) for representing the system
   - Three types of nodes:
     * Input nodes (environment measurements)
     * Hidden nodes (internal processing)
     * Output nodes (control decisions)
   - Progressive complexity approach: starting with minimal parameters and expanding as needed

2. **Performance Considerations**
   - Parameter optimization is a key challenge
   - System architecture evolves during operation
   - Clear separation between input, processing, and output stages

These insights inform our classical control implementation, particularly in:
- Structuring our state management to clearly separate inputs, processing, and outputs
- Starting with minimal control parameters and allowing for extensibility
- Designing our testing framework to validate each stage independently

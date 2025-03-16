# done
* basic parts in place
* [x] fetch and document double pendulum video reference
    * Created structured documentation in docs/references/double_pendulum.md
    * Set up transcript file double_pendulum_video.en.srt
    * Aligned with existing documentation standards
* [x] Clean up video transcripts for better reference
    * Removed timestamp information except for key sections
    * Consolidated subtitle fragments into complete sentences
    * Added section headers for clarity
    * Preserved context for visual demonstrations

# now
* [ ] try to gain more insights into the training process and the neural networks
    * I want better insights into how the neural network is training
    * Focus on temporal difference learning implementation
    * Add comprehensive state persistence for training progress
* [ ] Implement progressive difficulty increase during training
    * Gradually increase the balance threshold as training progresses
    * Adapt the fitness function to push solutions toward desired behavior
    * Implement success metrics for performance evaluation

# next
* [ ] Enhance neural network architecture based on video insights
    * Implement DAG-based representation (Directed Acyclic Graphs)
    * Ensure proper three-node design (input, hidden, output layers)
    * Optimize network structure for control tasks
* [ ] Improve training methodology
    * Implement evolutionary architecture that adapts over generations
    * Add clear success metrics for performance evaluation
    * Develop adaptive learning rate based on success rate
* video summaries need cleanup especially if they're helping to feed context and determine features and implementation details
    * rerun all video summaries from subtitles, maybe add a processing layer of converting the srt to plain text, then summarize
    * can we discern anything from the video?
    * skim all markdown for anything valuable, trash them, and create new ones
        * even if there is anything useful in there, it's surrounded by so much noise, we should probably keep it tighter, more readable, more understandable, less technical jargon, more clear behavior 
        * this includes README.md, RULES, ARCHITECTURE.md, PROGRESS.MD, TRAINING.MD

# soon
* Implement training pipeline improvements:
    * Add momentum-based backpropagation
    * Enhance reward prioritization
    * Implement adaptive learning rate
    * Expand checkpoint system
* [ ] Implement visualization improvements:
    * Add network state visualization
    * Display real-time training metrics
    * Show learning progress indicators
    * Visualize weight matrix operations
package neural

// Node represents a neuron in the network with its connections and processing logic
type Node struct {
	ID        int
	Bias      float64
	Value     float64
	Processed bool // For topological processing

	// Activation function - can be changed for different node types
	Activation func(float64) float64

	// Connections
	Incoming []*Connection
	Outgoing []*Connection
}

// Connection represents a weighted connection between two nodes
type Connection struct {
	From       *Node
	To         *Node
	Weight     float64
	Enabled    bool
	Innovation int // Unique historical marking for NEAT
}

// Network represents the complete neural network structure
type Network struct {
	Nodes       []*Node
	Connections []*Connection
	InputNodes  []*Node
	OutputNodes []*Node
	SortedNodes []*Node // Cached topological sort

	// Evolution tracking
	Generation  int
	Innovation  int // Global innovation counter
	Fitness     float64
}

// Population manages a collection of networks for evolution
type Population struct {
	Networks    []*Network
	Generation  int
	TopPercent  float64 // Percentage of top performers to keep
	MaxSize     int     // Maximum population size
}

// NetworkState represents the serializable state of the network
type NetworkState struct {
	// Node states
	NodeStates []struct {
		ID        int
		Bias      float64
		Value     float64
	}

	// Connection states
	ConnectionStates []struct {
		FromID     int
		ToID       int
		Weight     float64
		Enabled    bool
		Innovation int
	}

	// Network metadata
	Generation  int
	Innovation  int
	Fitness     float64
}

// NewNode creates a new node with the given ID and activation function
func NewNode(id int, activation func(float64) float64) *Node {
	return &Node{
		ID:         id,
		Activation: activation,
		Incoming:   make([]*Connection, 0),
		Outgoing:   make([]*Connection, 0),
	}
}

// NewConnection creates a new connection between two nodes
func NewConnection(from, to *Node, weight float64, innovation int) *Connection {
	conn := &Connection{
		From:       from,
		To:         to,
		Weight:     weight,
		Enabled:    true,
		Innovation: innovation,
	}

	// Add to node's connection lists
	from.Outgoing = append(from.Outgoing, conn)
	to.Incoming = append(to.Incoming, conn)

	return conn
}

// Process computes the node's output value
func (n *Node) Process() {
	if n.Processed {
		return
	}

	// Process incoming nodes first
	sum := n.Bias
	for _, conn := range n.Incoming {
		if conn.Enabled {
			sum += conn.From.Value * conn.Weight
		}
	}

	n.Value = n.Activation(sum)
	n.Processed = true
}

// Reset prepares the node for the next forward pass
func (n *Node) Reset() {
	n.Value = 0
	n.Processed = false
}

// Forward performs a forward pass through the network
func (net *Network) Forward(inputs []float64) []float64 {
	// Validate input size
	if len(inputs) != len(net.InputNodes) {
		panic("input size mismatch")
	}

	// Reset all nodes
	for _, node := range net.Nodes {
		node.Reset()
	}

	// Set input values
	for i, node := range net.InputNodes {
		node.Value = inputs[i]
		node.Processed = true
	}

	// Process nodes in topological order
	for _, node := range net.SortedNodes {
		node.Process()
	}

	// Collect outputs
	outputs := make([]float64, len(net.OutputNodes))
	for i, node := range net.OutputNodes {
		outputs[i] = node.Value
	}

	return outputs
}

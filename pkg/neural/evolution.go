package neural

import (
	"math"
	"math/rand"
	"sort"
)

// MutationRates defines probabilities for different mutation types
type MutationRates struct {
	NewConnection    float64
	SplitConnection float64
	ModifyWeight    float64
	ModifyBias      float64
}

// DefaultMutationRates returns standard mutation rates
func DefaultMutationRates() MutationRates {
	return MutationRates{
		NewConnection:    0.1,
		SplitConnection: 0.05,
		ModifyWeight:    0.3,
		ModifyBias:      0.2,
	}
}

// NewPopulation creates a new population with the specified size
func NewPopulation(size int, topPercent float64) *Population {
	return &Population{
		Networks:    make([]*Network, 0, size),
		Generation:  0,
		TopPercent:  topPercent,
		MaxSize:     size,
	}
}

// AddNetwork adds a network to the population
func (p *Population) AddNetwork(net *Network) {
	if len(p.Networks) < p.MaxSize {
		p.Networks = append(p.Networks, net)
	}
}

// Evolve performs one generation of evolution
func (p *Population) Evolve(rates MutationRates) {
	if len(p.Networks) == 0 {
		return
	}

	// Sort networks by fitness
	sort.Slice(p.Networks, func(i, j int) bool {
		return p.Networks[i].Fitness > p.Networks[j].Fitness
	})

	// Calculate how many top performers to keep
	topCount := int(float64(len(p.Networks)) * p.TopPercent)
	if topCount < 1 {
		topCount = 1
	}

	// Keep top performers
	topPerformers := p.Networks[:topCount]

	// Create new population starting with top performers
	newPopulation := make([]*Network, 0, p.MaxSize)
	newPopulation = append(newPopulation, topPerformers...)

	// Fill rest of population with mutated versions of top performers
	for len(newPopulation) < p.MaxSize {
		// Select parent from top performers
		parent := topPerformers[rand.Intn(len(topPerformers))]
		
		// Create offspring through mutation
		offspring := parent.Clone()
		offspring.Mutate(rates)
		
		newPopulation = append(newPopulation, offspring)
	}

	p.Networks = newPopulation
	p.Generation++
}

// Clone creates a deep copy of the network
func (net *Network) Clone() *Network {
	// Create new network
	clone := &Network{
		Generation: net.Generation,
		Innovation: net.Innovation,
		Nodes:      make([]*Node, len(net.Nodes)),
		InputNodes: make([]*Node, len(net.InputNodes)),
		OutputNodes: make([]*Node, len(net.OutputNodes)),
	}

	// Clone nodes
	nodeMap := make(map[*Node]*Node)
	for i, node := range net.Nodes {
		clone.Nodes[i] = NewNode(node.ID, node.Activation)
		clone.Nodes[i].Bias = node.Bias
		nodeMap[node] = clone.Nodes[i]
	}

	// Map input/output nodes
	for i, node := range net.InputNodes {
		clone.InputNodes[i] = nodeMap[node]
	}
	for i, node := range net.OutputNodes {
		clone.OutputNodes[i] = nodeMap[node]
	}

	// Clone connections
	clone.Connections = make([]*Connection, len(net.Connections))
	for i, conn := range net.Connections {
		clone.Connections[i] = NewConnection(
			nodeMap[conn.From],
			nodeMap[conn.To],
			conn.Weight,
			conn.Innovation,
		)
		clone.Connections[i].Enabled = conn.Enabled
	}

	return clone
}

// Mutate applies random mutations based on the given rates
func (net *Network) Mutate(rates MutationRates) {
	// Attempt each mutation type based on rates
	if rand.Float64() < rates.NewConnection {
		net.addRandomConnection()
	}
	if rand.Float64() < rates.SplitConnection {
		net.splitRandomConnection()
	}
	if rand.Float64() < rates.ModifyWeight {
		net.modifyRandomWeight()
	}
	if rand.Float64() < rates.ModifyBias {
		net.modifyRandomBias()
	}
}

// addRandomConnection adds a new connection between unconnected nodes
func (net *Network) addRandomConnection() {
	// Find all possible node pairs that could be connected
	var possiblePairs [][2]*Node
	for _, from := range net.Nodes {
		for _, to := range net.Nodes {
			// Skip if connection would create cycle or already exists
			if from != to && !net.hasConnection(from, to) && !net.wouldCreateCycle(from, to) {
				possiblePairs = append(possiblePairs, [2]*Node{from, to})
			}
		}
	}

	if len(possiblePairs) == 0 {
		return
	}

	// Select random pair and create connection
	pair := possiblePairs[rand.Intn(len(possiblePairs))]
	net.Innovation++
	conn := NewConnection(pair[0], pair[1], rand.Float64()*2-1, net.Innovation)
	net.Connections = append(net.Connections, conn)
}

// splitRandomConnection splits an existing connection and adds a new node
func (net *Network) splitRandomConnection() {
	enabledConns := make([]*Connection, 0)
	for _, conn := range net.Connections {
		if conn.Enabled {
			enabledConns = append(enabledConns, conn)
		}
	}

	if len(enabledConns) == 0 {
		return
	}

	// Select random connection to split
	conn := enabledConns[rand.Intn(len(enabledConns))]
	conn.Enabled = false

	// Create new node
	newNodeID := len(net.Nodes)
	newNode := NewNode(newNodeID, conn.From.Activation)
	net.Nodes = append(net.Nodes, newNode)

	// Create two new connections
	net.Innovation++
	conn1 := NewConnection(conn.From, newNode, 1.0, net.Innovation)
	net.Innovation++
	conn2 := NewConnection(newNode, conn.To, conn.Weight, net.Innovation)

	net.Connections = append(net.Connections, conn1, conn2)
}

// modifyRandomWeight slightly changes a random connection weight
func (net *Network) modifyRandomWeight() {
	if len(net.Connections) == 0 {
		return
	}

	conn := net.Connections[rand.Intn(len(net.Connections))]
	// Perturb weight by up to ±0.5
	conn.Weight += (rand.Float64() - 0.5)
	// Clip weight to reasonable range
	conn.Weight = math.Max(-2.0, math.Min(2.0, conn.Weight))
}

// modifyRandomBias slightly changes a random node's bias
func (net *Network) modifyRandomBias() {
	if len(net.Nodes) == 0 {
		return
	}

	node := net.Nodes[rand.Intn(len(net.Nodes))]
	// Perturb bias by up to ±0.5
	node.Bias += (rand.Float64() - 0.5)
	// Clip bias to reasonable range
	node.Bias = math.Max(-2.0, math.Min(2.0, node.Bias))
}

// hasConnection checks if a connection exists between two nodes
func (net *Network) hasConnection(from, to *Node) bool {
	for _, conn := range net.Connections {
		if conn.From == from && conn.To == to {
			return true
		}
	}
	return false
}

// wouldCreateCycle checks if adding a connection would create a cycle
func (net *Network) wouldCreateCycle(from, to *Node) bool {
	// Simple DFS to check for cycles
	visited := make(map[*Node]bool)
	var visit func(*Node) bool

	visit = func(node *Node) bool {
		if node == from {
			return true // Found cycle
		}
		if visited[node] {
			return false
		}
		visited[node] = true
		for _, conn := range node.Incoming {
			if conn.Enabled && visit(conn.From) {
				return true
			}
		}
		return false
	}

	return visit(to)
}

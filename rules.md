# Tech Stack Rules

## Go Tools
- Use `gofmt`, `goimports`, `govet`, `golint`, `golangci-lint`
- Go 1.21+ required

## Neural Network Implementation
- Three-node architecture (input, hidden, output)
- DAG-based representation
- Temporal difference learning

## Angle Conventions
- All angles normalized to [0, 2π]
- Measured clockwise from upward vertical
- 0 = upward pointing
- π = downward pointing
- π/2 = rightward pointing
- 3π/2 = leftward pointing

## Visualization
- Ebiten v2.8.6
- Debug overlay required
- Network state visualization

## Package Management
- Local `.packages` directory
- Set GOPATH to project's `.packages` when installing:
  ```
  GOPATH=/path/to/project/.packages go get <package>
  ```

## Documentation
- Update ARCHITECTURE.md for design decisions
- Maintain PROGRESS.md for status tracking
- Document API with examples

## Progress Tracking
- Current status in TODO.md
- Completed components
- In-progress features
- Next steps and priorities

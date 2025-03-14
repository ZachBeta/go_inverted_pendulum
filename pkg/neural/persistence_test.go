package neural

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestNetworkPersistence(t *testing.T) {
	// Create temp directory for test files
	tmpDir, err := os.MkdirTemp("", "network_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create test network with known weights
	net := NewNetwork()
	net.SetDebug(false) // Disable debug output during tests
	testWeights := []float64{1.5, -0.8, 0.3}
	testLearningRate := 0.1

	if err := net.SetWeights(testWeights); err != nil {
		t.Fatalf("Failed to set test weights: %v", err)
	}
	net.SetLearningRate(testLearningRate)

	t.Run("SaveAndLoad", func(t *testing.T) {
		savePath := filepath.Join(tmpDir, "test_network.json")

		// Save network state
		if err := net.SaveToFile(savePath); err != nil {
			t.Fatalf("Failed to save network: %v", err)
		}

		// Verify file exists and contains valid JSON
		data, err := os.ReadFile(savePath)
		if err != nil {
			t.Fatalf("Failed to read saved file: %v", err)
		}

		var state NetworkState
		if err := json.Unmarshal(data, &state); err != nil {
			t.Fatalf("Failed to parse saved JSON: %v", err)
		}

		// Create new network and load state
		loadedNet := NewNetwork()
		loadedNet.SetDebug(false)
		if err := loadedNet.LoadFromFile(savePath); err != nil {
			t.Fatalf("Failed to load network: %v", err)
		}

		// Compare weights
		loadedWeights := loadedNet.GetWeights()
		for i, w := range testWeights {
			if w != loadedWeights[i] {
				t.Errorf("Weight mismatch at index %d: want %.4f, got %.4f",
					i, w, loadedWeights[i])
			}
		}
	})

	t.Run("InvalidFile", func(t *testing.T) {
		// Test loading from non-existent file
		if err := net.LoadFromFile("/nonexistent/path"); err == nil {
			t.Error("Expected error when loading from non-existent file")
		}

		// Test loading invalid JSON
		invalidPath := filepath.Join(tmpDir, "invalid.json")
		if err := os.WriteFile(invalidPath, []byte("invalid json"), 0644); err != nil {
			t.Fatalf("Failed to create invalid JSON file: %v", err)
		}

		if err := net.LoadFromFile(invalidPath); err == nil {
			t.Error("Expected error when loading invalid JSON")
		}
	})

	t.Run("NestedDirectory", func(t *testing.T) {
		// Test saving to nested directory path
		nestedPath := filepath.Join(tmpDir, "nested", "dir", "network.json")
		if err := net.SaveToFile(nestedPath); err != nil {
			t.Fatalf("Failed to save to nested path: %v", err)
		}

		// Verify file exists
		if _, err := os.Stat(nestedPath); os.IsNotExist(err) {
			t.Error("Expected nested directory and file to be created")
		}
	})
}

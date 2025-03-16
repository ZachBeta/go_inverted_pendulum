// Package logger provides a custom logging system for the inverted pendulum application
// with support for different log levels and file output
package logger

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"
)

// LogLevel represents the severity of a log message
type LogLevel int

const (
	// DEBUG level for detailed information, typically only valuable for debugging
	DEBUG LogLevel = iota
	// INFO level for general operational information
	INFO
	// ERROR level for error events that might still allow the application to continue
	ERROR
	// NONE disables all logging
	NONE
)

// String returns the string representation of the log level
func (l LogLevel) String() string {
	switch l {
	case DEBUG:
		return "DEBUG"
	case INFO:
		return "INFO"
	case ERROR:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Logger is a custom logger with support for log levels and file output
type Logger struct {
	consoleLogger *log.Logger
	fileLogger    *log.Logger
	consoleLevel  LogLevel
	fileLevel     LogLevel
	file          *os.File
}

// NewLogger creates a new logger with the specified console and file log levels
func NewLogger(consoleLevel, fileLevel LogLevel) (*Logger, error) {
	// Create console logger
	consoleLogger := log.New(os.Stdout, "", log.LstdFlags)

	// Create logs directory if it doesn't exist
	logsDir := "logs"
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create logs directory: %v", err)
	}

	// Create log file with timestamp in name
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	logFilePath := filepath.Join(logsDir, fmt.Sprintf("pendulum_%s.log", timestamp))
	file, err := os.Create(logFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %v", err)
	}

	// Create file logger
	fileLogger := log.New(file, "", log.LstdFlags)

	return &Logger{
		consoleLogger: consoleLogger,
		fileLogger:    fileLogger,
		consoleLevel:  consoleLevel,
		fileLevel:     fileLevel,
		file:          file,
	}, nil
}

// Close closes the log file
func (l *Logger) Close() error {
	if l.file != nil {
		return l.file.Close()
	}
	return nil
}

// Debug logs a debug message
func (l *Logger) Debug(format string, v ...interface{}) {
	l.log(DEBUG, format, v...)
}

// Info logs an info message
func (l *Logger) Info(format string, v ...interface{}) {
	l.log(INFO, format, v...)
}

// Error logs an error message
func (l *Logger) Error(format string, v ...interface{}) {
	l.log(ERROR, format, v...)
}

// Fatal logs an error message and exits the program
func (l *Logger) Fatal(format string, v ...interface{}) {
	l.log(ERROR, format, v...)
	os.Exit(1)
}

// Printf provides compatibility with the standard log.Logger interface
func (l *Logger) Printf(format string, v ...interface{}) {
	l.Info(format, v...)
}

// Println provides compatibility with the standard log.Logger interface
func (l *Logger) Println(v ...interface{}) {
	l.Info("%v", fmt.Sprintln(v...))
}

// Fatalf provides compatibility with the standard log.Logger interface
func (l *Logger) Fatalf(format string, v ...interface{}) {
	l.Fatal(format, v...)
}

// log logs a message with the specified level
func (l *Logger) log(level LogLevel, format string, v ...interface{}) {
	// Format message with level prefix
	message := fmt.Sprintf("[%s] %s", level.String(), format)
	
	// Log to console if level is high enough
	if level >= l.consoleLevel {
		l.consoleLogger.Printf(message, v...)
	}
	
	// Log to file if level is high enough
	if level >= l.fileLevel {
		l.fileLogger.Printf(message, v...)
	}
}

// GetStandardLogger returns a standard log.Logger that writes to both console and file
// This is useful for compatibility with libraries that expect a standard logger
func (l *Logger) GetStandardLogger() *log.Logger {
	// Create a multi-writer that writes to both console and file
	multiWriter := io.MultiWriter(os.Stdout, l.file)
	return log.New(multiWriter, "", log.LstdFlags)
}

// SetConsoleLevel sets the minimum log level for console output
func (l *Logger) SetConsoleLevel(level LogLevel) {
	l.consoleLevel = level
}

// SetFileLevel sets the minimum log level for file output
func (l *Logger) SetFileLevel(level LogLevel) {
	l.fileLevel = level
}

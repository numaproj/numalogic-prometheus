package log

import (
	"go.uber.org/zap"
	"os"
)

// NewLogger returns a new zap.SugaredLogger
func NewLogger(name string) *zap.SugaredLogger {
	var config zap.Config
	debugMode, ok := os.LookupEnv("LOG_DEBUG")
	if ok && debugMode == "true" {
		config = zap.NewDevelopmentConfig()
	} else {
		config = zap.NewProductionConfig()
	}
	// Config customization goes here if any
	config.OutputPaths = []string{"stdout"}
	logger, err := config.Build()
	if err != nil {
		panic(err)
	}
	return logger.Named(name).Sugar()
}

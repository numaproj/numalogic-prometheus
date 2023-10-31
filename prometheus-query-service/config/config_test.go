package config

import (
	"github.com/stretchr/testify/assert"
	"gopkg.in/yaml.v3"
	"testing"
	"time"
)

var configYAML = `general:
  channelCapacity: 
    capacity: 20
    dropDisabled: false       
  maxAge: 60
prometheusJobs:
  - name: "query-1"
    queryConfig:
      interval: "20s"
      source: "http://localhost:9090"
      query: "namespace_app_rollouts_http_request_error_rate"
      backoff:
        durationSeconds: 1
        factor: 2
        maxSteps: 5
    sendConfig:
      targets:
        - url: "https://localhost:8443/vertices/input"
          insecure: true
      backoff:
        durationSeconds: 1
        factor: 2
        maxSteps: 5
`

func TestConfig(t *testing.T) {
	var config Config
	err := yaml.Unmarshal([]byte(configYAML), &config)
	assert.NoError(t, err)
	assert.NotNil(t, config.GeneralConfig.ChannelCapacity)
	assert.Equal(t, 20, config.GeneralConfig.ChannelCapacity.Capacity)
	assert.Equal(t, 20, config.GeneralConfig.ChannelCapacity.GetCapacity())
	assert.Equal(t, false, config.GeneralConfig.ChannelCapacity.DropDisabled)

	assert.Equal(t, 1, len(config.PrometheusJobs))
	assert.NotNil(t, config.PrometheusJobs[0].QueryConfig)
	assert.Equal(t, "20s", config.PrometheusJobs[0].QueryConfig.Interval)
	assert.Equal(t, "namespace_app_rollouts_http_request_error_rate", config.PrometheusJobs[0].QueryConfig.Query)
	assert.Equal(t, "http://localhost:9090", config.PrometheusJobs[0].QueryConfig.Source)
	assert.NotNil(t, config.PrometheusJobs[0].QueryConfig.Backoff)
	assert.NotNil(t, config.PrometheusJobs[0].QueryConfig.Backoff.Get())
	assert.Equal(t, float64(2), config.PrometheusJobs[0].QueryConfig.Backoff.Factor)
	assert.Equal(t, "http://localhost:9090", config.PrometheusJobs[0].QueryConfig.Source)
	assert.NotNil(t, config.PrometheusJobs[0].SendConfig)
	assert.NotNil(t, config.PrometheusJobs[0].SendConfig.Backoff)
	assert.Equal(t, float64(2), config.PrometheusJobs[0].SendConfig.Backoff.Factor)
	dur, err := config.PrometheusJobs[0].QueryConfig.GetInterval()
	assert.NoError(t, err)
	assert.Equal(t, 20*time.Second, *dur)
}

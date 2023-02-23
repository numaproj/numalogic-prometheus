package main

import (
	"encoding/json"
	"github.com/prometheus/prometheus/prompb"
	"github.com/stretchr/testify/assert"
	"testing"
	"time"
)

func TestProcessPrometheusData(t *testing.T) {
	req := prompb.WriteRequest{
		Timeseries: []prompb.TimeSeries{{
			Labels: []prompb.Label{
				{Name: "__name__", Value: "go_gc_duration_seconds"},
				{Name: "instance", Value: "localhost:9090"},
				{Name: "job", Value: "prometheus"},
				{Name: "quantile", Value: "0.99"},
			},

			Samples: []prompb.Sample{
				{Value: 4.63, Timestamp: time.Now().UnixNano()},
			},
		},
		},
	}

	result, err := processPrometheusData(&req)
	var metrics map[string]interface{}
	err = json.Unmarshal(result[0], &metrics)
	assert.NoError(t, err)
	assert.Len(t, metrics, 4)
	assert.Equal(t, "go_gc_duration_seconds", metrics["name"])
	assert.Equal(t, string("4.63"), metrics["value"])
}

package main

import (
	"encoding/json"
	"github.com/prometheus/prometheus/prompb"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
	"time"
)

func TestProcessPrometheusData(t *testing.T) {
	t.Run("Check float value", func(t *testing.T) {
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

		result, err := processPrometheusData("123", &req)
		var metrics map[string]interface{}
		err = json.Unmarshal(result[0], &metrics)
		assert.NoError(t, err)
		assert.Len(t, metrics, 4)
		assert.Equal(t, "go_gc_duration_seconds", metrics["name"])
		assert.Equal(t, string("4.63"), metrics["value"])
	},
	)
	t.Run("Check NaN value", func(t *testing.T) {
		req := prompb.WriteRequest{
			Timeseries: []prompb.TimeSeries{{
				Labels: []prompb.Label{
					{Name: "__name__", Value: "go_gc_duration_seconds"},
					{Name: "instance", Value: "localhost:9090"},
					{Name: "job", Value: "prometheus"},
					{Name: "quantile", Value: "0.99"},
				},

				Samples: []prompb.Sample{
					{Value: math.NaN(), Timestamp: time.Now().UnixNano()},
				},
			},
			},
		}

		result, err := processPrometheusData("123", &req)
		var metrics map[string]interface{}
		err = json.Unmarshal(result[0], &metrics)
		assert.NoError(t, err)
		assert.Len(t, metrics, 4)
		assert.Equal(t, "go_gc_duration_seconds", metrics["name"])
		assert.Equal(t, string("NaN"), metrics["value"])
	},
	)
	t.Run("Check +Inf value", func(t *testing.T) {
		req := prompb.WriteRequest{
			Timeseries: []prompb.TimeSeries{{
				Labels: []prompb.Label{
					{Name: "__name__", Value: "go_gc_duration_seconds"},
					{Name: "instance", Value: "localhost:9090"},
					{Name: "job", Value: "prometheus"},
					{Name: "quantile", Value: "0.99"},
				},

				Samples: []prompb.Sample{
					{Value: math.Inf(0), Timestamp: time.Now().UnixNano()},
				},
			},
			},
		}

		result, err := processPrometheusData("123", &req)
		var metrics map[string]interface{}
		err = json.Unmarshal(result[0], &metrics)
		assert.NoError(t, err)
		assert.Len(t, metrics, 4)
		assert.Equal(t, "go_gc_duration_seconds", metrics["name"])
		assert.Equal(t, string("+Inf"), metrics["value"])
	},
	)
	t.Run("Check -Inf value", func(t *testing.T) {
		req := prompb.WriteRequest{
			Timeseries: []prompb.TimeSeries{{
				Labels: []prompb.Label{
					{Name: "__name__", Value: "go_gc_duration_seconds"},
					{Name: "instance", Value: "localhost:9090"},
					{Name: "job", Value: "prometheus"},
					{Name: "quantile", Value: "0.99"},
				},

				Samples: []prompb.Sample{
					{Value: math.Inf(-1), Timestamp: time.Now().UnixNano()},
				},
			},
			},
		}

		result, err := processPrometheusData("123", &req)
		var metrics map[string]interface{}
		err = json.Unmarshal(result[0], &metrics)
		assert.NoError(t, err)
		assert.Len(t, metrics, 4)
		assert.Equal(t, "go_gc_duration_seconds", metrics["name"])
		assert.Equal(t, string("-Inf"), metrics["value"])
	},
	)

}

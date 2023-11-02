package prometheus

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/config"
	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/pkg/log"
	prometheusV1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"

	"github.com/stretchr/testify/assert"
)

type MockPrometheusAPI struct {
}

func (mpa *MockPrometheusAPI) Alerts(ctx context.Context) (prometheusV1.AlertsResult, error) {
	return prometheusV1.AlertsResult{}, nil
}
func (mpa *MockPrometheusAPI) AlertManagers(ctx context.Context) (prometheusV1.AlertManagersResult, error) {
	return prometheusV1.AlertManagersResult{}, nil
}
func (mpa *MockPrometheusAPI) CleanTombstones(ctx context.Context) error { return nil }
func (mpa *MockPrometheusAPI) Config(ctx context.Context) (prometheusV1.ConfigResult, error) {
	return prometheusV1.ConfigResult{}, nil
}
func (mpa *MockPrometheusAPI) DeleteSeries(ctx context.Context, matches []string, startTime, endTime time.Time) error {
	return nil
}
func (mpa *MockPrometheusAPI) Flags(ctx context.Context) (prometheusV1.FlagsResult, error) {
	return prometheusV1.FlagsResult{}, nil
}
func (mpa *MockPrometheusAPI) LabelNames(ctx context.Context, matches []string, startTime, endTime time.Time) ([]string, prometheusV1.Warnings, error) {
	return []string{}, prometheusV1.Warnings{}, nil
}
func (mpa *MockPrometheusAPI) LabelValues(ctx context.Context, label string, matches []string, startTime, endTime time.Time) (model.LabelValues, prometheusV1.Warnings, error) {
	return model.LabelValues{}, prometheusV1.Warnings{}, nil
}

var queryIndex int

func (mpa *MockPrometheusAPI) Query(ctx context.Context, query string, ts time.Time, opts ...prometheusV1.Option) (model.Value, prometheusV1.Warnings, error) {

	switch query {
	case "my_scalar":
		return scalarValue, prometheusV1.Warnings{}, nil
	case "my_vector":
		queryIndex++
		fmt.Println(queryIndex)
		switch queryIndex {
		case 1, 2:
			return model.Vector([]*model.Sample{}), prometheusV1.Warnings{}, &prometheusV1.Error{Type: prometheusV1.ErrTimeout}
		case 3:
			return vectorValue, prometheusV1.Warnings{}, nil
		}

		return model.Vector([]*model.Sample{}),
			prometheusV1.Warnings{},
			nil

	}

	return MockPrometheusValue{}, prometheusV1.Warnings{}, nil
}
func (mpa *MockPrometheusAPI) QueryRange(ctx context.Context, query string, r prometheusV1.Range, opts ...prometheusV1.Option) (model.Value, prometheusV1.Warnings, error) {
	return MockPrometheusValue{}, prometheusV1.Warnings{}, nil
}
func (mpa *MockPrometheusAPI) QueryExemplars(ctx context.Context, query string, startTime, endTime time.Time) ([]prometheusV1.ExemplarQueryResult, error) {
	return []prometheusV1.ExemplarQueryResult{}, nil
}
func (mpa *MockPrometheusAPI) Buildinfo(ctx context.Context) (prometheusV1.BuildinfoResult, error) {
	return prometheusV1.BuildinfoResult{}, nil
}
func (mpa *MockPrometheusAPI) Runtimeinfo(ctx context.Context) (prometheusV1.RuntimeinfoResult, error) {
	return prometheusV1.RuntimeinfoResult{}, nil
}
func (mpa *MockPrometheusAPI) Series(ctx context.Context, matches []string, startTime, endTime time.Time) ([]model.LabelSet, prometheusV1.Warnings, error) {
	return make([]model.LabelSet, 0), prometheusV1.Warnings{}, nil
}
func (mpa *MockPrometheusAPI) Snapshot(ctx context.Context, skipHead bool) (prometheusV1.SnapshotResult, error) {
	return prometheusV1.SnapshotResult{}, nil
}
func (mpa *MockPrometheusAPI) Rules(ctx context.Context) (prometheusV1.RulesResult, error) {
	return prometheusV1.RulesResult{}, nil
}
func (mpa *MockPrometheusAPI) Targets(ctx context.Context) (prometheusV1.TargetsResult, error) {
	return prometheusV1.TargetsResult{}, nil
}
func (mpa *MockPrometheusAPI) TargetsMetadata(ctx context.Context, matchTarget, metric, limit string) ([]prometheusV1.MetricMetadata, error) {
	return []prometheusV1.MetricMetadata{}, nil
}
func (mpa *MockPrometheusAPI) Metadata(ctx context.Context, metric, limit string) (map[string][]prometheusV1.Metadata, error) {
	return map[string][]prometheusV1.Metadata{}, nil
}
func (mpa *MockPrometheusAPI) TSDB(ctx context.Context) (prometheusV1.TSDBResult, error) {
	return prometheusV1.TSDBResult{}, nil
}
func (mpa *MockPrometheusAPI) WalReplay(ctx context.Context) (prometheusV1.WalReplayStatus, error) {
	return prometheusV1.WalReplayStatus{}, nil
}

type MockPrometheusValue struct {
}

func (mpv MockPrometheusValue) Type() model.ValueType {
	return model.ValNone
}
func (mpv MockPrometheusValue) String() string {
	return "novalue"
}

var mockPrometheusAPI MockPrometheusAPI

var scalarValue = &(model.Scalar{Value: model.SampleValue(123), Timestamp: model.TimeFromUnix(1684444154)})
var vectorValue = model.Vector([]*model.Sample{
	{
		Metric:    model.Metric(model.LabelSet{"my-key-1": "my-value-1", "my-key-2": "my-value-2"}),
		Value:     model.SampleValue(123),
		Timestamp: model.TimeFromUnix(1684444154),
	},
	{
		Metric:    model.Metric(model.LabelSet{}),
		Value:     model.SampleValue(456),
		Timestamp: model.TimeFromUnix(1684444155),
	},
})
var vector0Value = model.Vector([]*model.Sample{})

// use mockPrometheusAPI to mock PrometheusAPI.Query() - which can return various errors
// test backoff strategies
func TestSendQuery(t *testing.T) {
	logger := log.NewLogger("test")

	// mock PrometheusAPI.Query() is set to fail the first 2 times and then succeed

	qe := NewQueryExecutor(
		make(chan config.Message, 5),
		"my-job",
		&config.QueryConfig{
			Interval: "30s",
			Source:   "http://prometheus:9090",
			Query:    "my_vector",
			Backoff: &config.BackoffStrategy{
				DurationSeconds: 1,
				Factor:          2,
				MaxSteps:        5,
			},
		},
		&config.GeneralConfig{
			ChannelCapacity: &config.ChannelCapacity{Capacity: 10},
		},
		"my_metric",
		logger,
	)
	qe.prometheusAPI = &mockPrometheusAPI

	tests := []struct {
		name            string
		backoffMaxSteps int
		expectedErr     bool
	}{
		{
			"fail",
			2,
			true,
		},
		{
			"succeed",
			3,
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			queryIndex = 0
			qe.queryConfig.Backoff.MaxSteps = tt.backoffMaxSteps
			_, err := qe.sendQuery(context.TODO(), logger, time.Now())
			if tt.expectedErr {
				assert.NotNil(t, err)
			} else {
				assert.Nil(t, err)
			}
		})
	}

}

type msgMap map[string]interface{}

// test:
// - Scalar and Vector
// - Vector can have multiple values, or 0 values
func TestProcessPrometheusMsg(t *testing.T) {
	logger := log.NewLogger("test")

	qe := NewQueryExecutor(
		make(chan config.Message, 5),
		"my-job",
		&config.QueryConfig{
			Interval: "30s",
			Source:   "http://prometheus:9090",
			Query:    "my_metric",
			Backoff:  nil,
		},
		&config.GeneralConfig{
			ChannelCapacity: &config.ChannelCapacity{Capacity: 10},
		},
		"my_metric",
		logger,
	)

	tests := []struct {
		name            string
		prometheusValue model.Value
		expectedMsgs    []msgMap
	}{
		{
			"scalar",
			scalarValue,
			[]msgMap{
				{
					"timestamp": "1684444154000",
					"value":     "123",
				},
			},
		},
		{
			"vector",
			vectorValue,
			[]msgMap{
				{
					"timestamp": "1684444154000",
					"value":     "123",
					"name":      "my_metric",
					"labels": map[string]interface{}{
						"my-key-1": "my-value-1",
						"my-key-2": "my-value-2",
					},
				},
				{
					"timestamp": "1684444155000",
					"value":     "456",
					"name":      "my_metric",
					"labels":    map[string]interface{}{},
				},
			},
		},
		{
			"vector of 0 length",
			vector0Value,
			[]msgMap{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msgs, err := qe.processPrometheusMsg(time.Now(), tt.prometheusValue, logger)
			assert.Nil(t, err)
			assert.Equal(t, len(tt.expectedMsgs), len(msgs))

			var mmap msgMap

			for i, expectedMsg := range tt.expectedMsgs {
				err := json.Unmarshal([]byte(msgs[i].Content), &mmap)
				assert.Nil(t, err)
				if !reflect.DeepEqual(expectedMsg, mmap) {
					assert.Failf(t, "Messages don't match", "expected=%#v, actual=%#v", expectedMsg, mmap)
				}
			}
		})
	}

	msgs, err := qe.processPrometheusMsg(time.Now(), scalarValue, logger)
	assert.Nil(t, err)
	assert.Equal(t, 1, len(msgs))
}

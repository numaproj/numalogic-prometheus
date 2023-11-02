package prometheus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/config"

	prometheusapi "github.com/prometheus/client_golang/api"
	prometheusV1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusModel "github.com/prometheus/common/model"
	"k8s.io/apimachinery/pkg/util/wait"

	"go.uber.org/zap"
)

type QueryExecutor struct {
	queue         chan config.Message
	log           *zap.SugaredLogger
	jobName       string
	queryConfig   *config.QueryConfig
	generalConfig *config.GeneralConfig
	metricName    string
	interval      time.Duration
	prometheusAPI prometheusV1.API
}

func NewQueryExecutor(queue chan config.Message, jobName string, queryConfig *config.QueryConfig, generalConfig *config.GeneralConfig, metricName string, log *zap.SugaredLogger) *QueryExecutor {
	prometheusAPI, err := createPrometheusAPI(queryConfig.Source)
	if err != nil {
		log.Fatal(err)
	}
	log.Infof("successfully connected to Prometheus at address %q", queryConfig.Source)
	interval, err := queryConfig.GetInterval()
	if err != nil {
		log.Fatal(err)
	}

	executor := QueryExecutor{
		log:           log,
		jobName:       jobName,
		queue:         queue,
		queryConfig:   queryConfig,
		generalConfig: generalConfig,
		metricName:    metricName,
		interval:      *interval,
		prometheusAPI: prometheusAPI,
	}

	return &executor

}

func createPrometheusAPI(address string) (prometheusV1.API, error) {

	prometheusApiConfig := prometheusapi.Config{
		Address: address,
	}
	client, err := prometheusapi.NewClient(prometheusApiConfig)
	if err != nil {
		return nil, err
	}
	return prometheusV1.NewAPI(client), nil
}

func (qe *QueryExecutor) Run(ctx context.Context) {
	qe.log.Debugf("Running Query executor at interval %v", qe.interval)
	ticker := time.NewTicker(qe.interval)
	for {
		select {
		case <-ticker.C:
			sendTime := time.Now()

			requestId := fmt.Sprintf("%s-%d", qe.jobName, sendTime.Unix())
			log := qe.log.With("request-id", requestId)

			// send query to Prometheus
			responses, err := qe.sendQuery(ctx, log, sendTime)
			if err != nil {
				log.Error(err)
				continue
			}

			// verify there's room in our channel before bothering to publish these messages
			// note: if capacity==0, this means there is no capacity limit
			capacitySettings := qe.generalConfig.ChannelCapacity
			if len(responses) > 0 && !capacitySettings.DropDisabled && len(qe.queue) == capacitySettings.GetCapacity() {
				// no room, drop message
				log.Warnf("QueryExecutor dropping message as we've reached capacity %d", cap(qe.queue))
				continue
			}
			// publish to Senders
			for _, response := range responses {
				log.Debugf("publishing: %+v, current channel length=%d", response, len(qe.queue))
				qe.queue <- response
			}
		case <-ctx.Done():
			return
		}
	}

}

// query Prometheus/Thanos and return response
func (qe *QueryExecutor) sendQuery(ctx context.Context, log *zap.SugaredLogger, dataTime time.Time) ([]config.Message, error) {

	log.Debugf("about to send query %q at time %v", qe.queryConfig.Query, dataTime)

	var response prometheusModel.Value
	var warnings prometheusV1.Warnings

	backoffStrategy := qe.queryConfig.Backoff.Get()
	if backoffStrategy.MaxSteps < 1 {
		backoffStrategy.MaxSteps = 1 // this value includes initial step, so "1" means don't retry
	}

	lastRetryTime := dataTime.Add(qe.interval).Add(-1 * time.Second) // at this point we need to give up because the next interval is starting

	backoff := wait.Backoff{
		Steps:    backoffStrategy.MaxSteps,
		Duration: time.Duration(backoffStrategy.DurationSeconds) * time.Second,
		Factor:   backoffStrategy.Factor,
	}
	retryIndex := 0

	retryError := wait.ExponentialBackoffWithContext(ctx, backoff, func(context.Context) (done bool, err error) {
		retryIndex++
		if time.Now().After(lastRetryTime) {
			return true, errors.New("failed all retried queries; next interval starting")
		}
		response, warnings, err = qe.prometheusAPI.Query(ctx, qe.queryConfig.Query, dataTime)
		if len(warnings) > 0 {
			for _, warning := range warnings {
				log.Warnf("Got warning in response to query: %q", warning)
			}
		}
		if err != nil {
			if possibleTransientError(err) {
				log.Debugf("error sending query (try %d): %v", retryIndex, err)
				return false, nil
			} else {
				return true, err // will log from caller
			}
		}

		return true, nil
	})
	if retryError != nil {
		return nil, retryError
	}

	jsonMsgs, err := qe.processPrometheusMsg(dataTime, response, log)
	if err != nil {
		return nil, err
	}

	return jsonMsgs, nil
}

func possibleTransientError(err error) bool {
	prometheusV1Error, ok := err.(*prometheusV1.Error)
	if !ok {
		return true
	}
	notTransient := prometheusV1Error.Type == prometheusV1.ErrBadData || prometheusV1Error.Type == prometheusV1.ErrBadResponse
	return !notTransient
}

func msecToTime(msec int64) time.Time {
	sec := msec / 1000
	nanosec := (msec % 1000) * 1000000
	return time.Unix(sec, nanosec)
}

// Process the message which may contain one or more data points from this moment in time
// (each representing a unique set of key/value pairs)
func (qe *QueryExecutor) processPrometheusMsg(queryTime time.Time, msg prometheusModel.Value, log *zap.SugaredLogger) ([]config.Message, error) {

	requestId := fmt.Sprintf("%s-%d", qe.jobName, queryTime.Unix())

	switch value := msg.(type) {
	// todo: is it useful to have Scalar? do we want to include "name" and "labels" keys like in Vector?
	case *prometheusModel.Scalar:
		sample := value
		log.Infof("QueryExecutor: got Scalar value back: %+v", sample)

		m := map[string]interface{}{
			"timestamp": strconv.FormatInt(int64(sample.Timestamp), 10),
			"value":     strconv.FormatFloat(float64(sample.Value), 'f', -1, 64),
		}
		data, err := json.Marshal(m)
		if err != nil {
			return nil, fmt.Errorf("error encountered while marshalling the time series: %v", zap.Error(err))
		}
		returnMsgs := []config.Message{
			{
				Content:   string(data),
				Timestamp: msecToTime(int64(sample.Timestamp)),
				RequestId: requestId,
			},
		}
		return returnMsgs, nil

	case prometheusModel.Vector:
		// Each of the results returned is a unique set of key/value pairs (labels)
		// Each of these can be turned into a unique message

		log.Infof("QueryExecutor: got Vector value back (length %d)", len(value))

		if len(value) == 0 {
			log.Warn("no data returned")
		}
		results := make([]config.Message, len(value))

		name := qe.metricName

		for i, sample := range value {

			labels := make(map[string]string, len(sample.Metric))
			for k, v := range sample.Metric {
				if k == prometheusModel.MetricNameLabel {
					if name == "" {
						name = string(v)
					}
				} else {
					labels[string(k)] = string(v)
				}
			}

			if name == "" {
				return nil, fmt.Errorf("metricName not set, and query returned no metric name either")
			}
			m := map[string]interface{}{
				"timestamp": strconv.FormatInt(int64(sample.Timestamp), 10),
				"value":     strconv.FormatFloat(float64(sample.Value), 'f', -1, 64),
				"name":      name,
				"labels":    labels,
			}
			data, err := json.Marshal(m)
			if err != nil {
				return nil, fmt.Errorf("error encountered while marshalling the time series: %v", zap.Error(err))
			}

			results[i] = config.Message{
				Content:   string(data),
				Timestamp: msecToTime(int64(sample.Timestamp)),
				RequestId: requestId,
			}
		}
		return results, nil

	default:
		return nil, fmt.Errorf("prometheus metric type not supported: %+v", msg)
	}
}

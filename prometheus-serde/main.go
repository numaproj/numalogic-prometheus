package main

import (
	"bytes"
	"context"
	"encoding/json"
	functionsdk "github.com/numaproj/numaflow-go/pkg/function"
	"github.com/numaproj/numaflow-go/pkg/function/server"
	logger "github.com/numaproj/numaflow/pkg/shared/logging"
	"github.com/prometheus/prometheus/prompb"
	"github.com/prometheus/prometheus/storage/remote"
	"go.uber.org/zap"
	"math"
	"strconv"
)

var log *zap.SugaredLogger

func processPrometheusData(req *prompb.WriteRequest) ([][]byte, error) {
	result := make([][]byte, 0)

	for _, ts := range req.Timeseries {
		labels := make(map[string]string, len(ts.Labels))

		for _, l := range ts.Labels {
			labels[l.Name] = l.Value
		}

		for _, sample := range ts.Samples {
			name := labels["__name__"]

			if math.IsNaN(sample.Value) {
				log.Infof("sample came with NaN value , we are ignoring.", sample)
				continue
			}

			epoch := sample.Timestamp
			m := map[string]interface{}{
				"timestamp": strconv.FormatInt(epoch, 10),
				"value":     strconv.FormatFloat(sample.Value, 'f', -1, 64),
				"name":      name,
				"labels":    labels,
			}

			data, err := json.Marshal(m)
			if err != nil {
				log.Error("error encountered while marshalling the time series", zap.Error(err))
			}
			result = append(result, data)
		}
	}

	return result, nil
}

func handle(ctx context.Context, key string, data functionsdk.Datum) functionsdk.Messages {
	req, err := remote.DecodeWriteRequest(bytes.NewReader(data.Value()))
	if err != nil {
		log.Errorf("Decode failed: %s", err)
		return nil
	}

	results, err := processPrometheusData(req)
	if err != nil {
		log.Errorf("Process failed: %s", err)
		return nil
	}

	mb := functionsdk.MessagesBuilder()
	for _, result := range results {
		log.Debugf("Payload: %s", string(result))

		mb = mb.Append(functionsdk.MessageToAll(result))
	}

	return mb
}

func main() {
	log = logger.NewLogger()
	server.New().RegisterMapper(functionsdk.MapFunc(handle)).Start(context.Background())
}

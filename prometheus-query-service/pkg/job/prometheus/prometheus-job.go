package prometheus

import (
	"context"

	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/config"
	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/pkg/log"
	zap "go.uber.org/zap"
)

type PrometheusJob struct {
	ctx           context.Context
	generalConfig *config.GeneralConfig
	config        config.QueryJob
	dataQueue     chan config.Message
	log           *zap.SugaredLogger
}

type QueryWorker interface {
	Run(ctx context.Context, queue chan string)
}

type SendWorker interface {
	Run(ctx context.Context, queue chan string)
}

func NewPrometheusJob(ctx context.Context, genConfig *config.GeneralConfig, jobConfig config.QueryJob) *PrometheusJob {
	pj := PrometheusJob{
		ctx:           ctx,
		config:        jobConfig,
		generalConfig: genConfig,
	}
	pj.log = log.NewLogger(jobConfig.Name).With("job", jobConfig)

	capacity := genConfig.ChannelCapacity.GetCapacity()
	pj.log.Infof("Creating channel from QueryExecutor to Sender with capacity %d", capacity)
	pj.dataQueue = make(chan config.Message, capacity)
	return &pj
}

func (pj *PrometheusJob) Run(ctx context.Context) error {
	executor := NewQueryExecutor(pj.dataQueue, pj.config.Name, pj.config.QueryConfig, pj.generalConfig, pj.config.MetricName, pj.log)
	sender := NewSender(pj.dataQueue, pj.config.SendConfig, pj.generalConfig, pj.log)
	pj.log.Info("create")
	go executor.Run(ctx)
	sender.Run(ctx)
	return nil
}

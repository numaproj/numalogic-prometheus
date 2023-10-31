package config

import (
	"time"
)

type Config struct {
	PrometheusJobs []QueryJob     `yaml:"prometheusJobs"`
	GeneralConfig  *GeneralConfig `yaml:"general,omitempty"`
}

type GeneralConfig struct {
	ChannelCapacity *ChannelCapacity `yaml:"channelCapacity,omitempty"`
	// drop message if older than this time
	// if set to 0, this behavior is disabled
	MaxAgeSeconds         int  `yaml:"maxAge,omitempty"`
	LeaderElectionEnabled bool `yaml:"leaderElectionEnabled,omitempty"`
}

type ChannelCapacity struct {
	// we don't allow 0 Capacity currently - 0 and undefined will both result in DefaultChannelCapacity
	Capacity int `yaml:"capacity,omitempty"`
	// if disabled, the publisher won't drop when capacity is full
	DropDisabled bool `yaml:"dropDisabled,omitempty"`
}

type QueryJob struct {
	// identifies this Job
	Name string `yaml:"name"`
	// identifies the name that we publish out; or if this isn't set, use the metric name that comes back from Prometheus
	MetricName  string       `yaml:"metricName"`
	QueryConfig *QueryConfig `yaml:"queryConfig"`
	SendConfig  *SendConfig  `yaml:"sendConfig"`
}

type QueryConfig struct {
	Interval string `yaml:"interval"`
	Source   string `yaml:"source"`
	Query    string `yaml:"query"`

	Backoff *BackoffStrategy `yaml:"backoff,omitempty"`
}

type SendConfig struct {
	Targets []Target         `yaml:"targets"`
	Backoff *BackoffStrategy `yaml:"backoff,omitempty"`
}

type BackoffStrategy struct {
	// DurationSeconds is the amount to back off (initially) in seconds
	DurationSeconds int `yaml:"durationSeconds,omitempty"`
	// Factor is a factor to multiply the base duration after each failed retry
	Factor float64 `yaml:"factor,omitempty"`
	// MaxSteps is 1 plus the maximum number of retries (i.e. includes the first one)
	MaxSteps int `yaml:"maxSteps,omitempty"`
}

var (
	DefaultBackoffDurationSeconds = 1
	DefaultBackoffFactor          = 2.0
	DefaultBackoffMaxSteps        = 3
	DefaultChannelCapacity        = 100
)

func (bs *BackoffStrategy) Get() *BackoffStrategy {
	backoffStrategy := &BackoffStrategy{
		DurationSeconds: DefaultBackoffDurationSeconds,
		Factor:          DefaultBackoffFactor,
		MaxSteps:        DefaultBackoffMaxSteps,
	}
	if bs == nil {
		return backoffStrategy
	}
	if bs.DurationSeconds != 0 {
		backoffStrategy.DurationSeconds = bs.DurationSeconds
	}
	if bs.Factor != 0 {
		backoffStrategy.Factor = bs.Factor
	}
	if bs.MaxSteps != 0 {
		backoffStrategy.MaxSteps = bs.MaxSteps
	}
	return backoffStrategy
}

func (q *QueryConfig) GetInterval() (*time.Duration, error) {
	dur, err := time.ParseDuration(q.Interval)
	if err != nil {
		return nil, err
	}
	return &dur, nil
}

type Target struct {
	Url      string `yaml:"url"`
	Insecure bool   `yaml:"insecure"`
}

func (cc *ChannelCapacity) GetCapacity() int {
	if cc.Capacity == 0 {
		return DefaultChannelCapacity
	}
	return cc.Capacity
}

type Message struct {
	Content   string
	Timestamp time.Time
	RequestId string
}

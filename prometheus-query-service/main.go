package main

import (
	"bufio"
	"context"
	"io"
	"k8s.io/client-go/rest"

	"os"
	"sync"
	"time"

	"go.uber.org/zap"
	"gopkg.in/yaml.v3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"

	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/config"
	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/pkg/job/prometheus"
	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/pkg/log"
)

func main() {

	logger := log.NewLogger("application")
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	config_path, ok := os.LookupEnv("CONFIG_PATH")
	if !ok {
		logger.Fatal("CONFIG_PATH env variable not found")
	}
	config, err := loadConfig(logger, config_path)
	if err != nil {
		logger.Fatal(err)
	}

	if config.GeneralConfig.LeaderElectionEnabled {
		logger.Info("starting with leader election")
		leaderName := "metrics-query-service"
		k8sConfig, err := rest.InClusterConfig()
		if err != nil {
			panic(err)
		}
		namespace := os.Getenv("POD_NAMESPACE")
		podName := os.Getenv("POD_NAME")
		logger.Info(namespace, podName)
		if err != nil {
			panic(err)
		}
		kubeClientSet := kubernetes.NewForConfigOrDie(k8sConfig)
		leaderelection.RunOrDie(ctx, leaderelection.LeaderElectionConfig{
			Lock: &resourcelock.LeaseLock{
				LeaseMeta: metav1.ObjectMeta{Name: leaderName, Namespace: namespace}, Client: kubeClientSet.CoordinationV1(),
				LockConfig: resourcelock.ResourceLockConfig{Identity: podName},
			},
			ReleaseOnCancel: false,
			LeaseDuration:   15 * time.Second,
			RenewDeadline:   10 * time.Second,
			RetryPeriod:     5 * time.Second,
			Callbacks: leaderelection.LeaderCallbacks{
				OnStartedLeading: func(ctx context.Context) {
					run(ctx, config)
				},
				OnStoppedLeading: func() {
					logger.Info("stopped leading")
					cancel()
				},
				OnNewLeader: func(identity string) {
					logger.With("leader", identity).Info("new leader")
				},
			},
		})

	} else {
		logger.Info("Server is starting without Leader Election")
		run(ctx, config)
	}
}

func run(ctx context.Context, config *config.Config) {
	wg := sync.WaitGroup{}
	for _, job := range config.PrometheusJobs {
		wg.Add(1)
		pj := prometheus.NewPrometheusJob(ctx, config.GeneralConfig, job)
		go pj.Run(ctx)
	}
	wg.Wait()
}

func loadConfig(log *zap.SugaredLogger, path string) (*config.Config, error) {
	var c config.Config
	file, err := os.Open(path)
	if err != nil {
		log.Error(err)
		return nil, err
	}

	defer file.Close()

	// Get the file size
	stat, err := file.Stat()
	if err != nil {
		log.Error(err)
		return nil, err
	}

	// Read the file into a byte slice
	bs := make([]byte, stat.Size())
	_, err = bufio.NewReader(file).Read(bs)
	if err != nil && err != io.EOF {
		log.Error(err)
		return nil, err
	}

	err = yaml.Unmarshal(bs, &c)
	log.Infof("Config: %+v\n", c)
	return &c, err
}

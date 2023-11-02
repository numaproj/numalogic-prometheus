package prometheus

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/numaproj/numalogic-prometheus/prometheus-query-service/config"
	"go.uber.org/zap"
	"k8s.io/apimachinery/pkg/util/wait"
)

type Sender struct {
	queue         chan config.Message
	log           *zap.SugaredLogger
	sendConfig    *config.SendConfig
	generalConfig *config.GeneralConfig
	chans         []chan config.Message
}

func NewSender(queue chan config.Message, sendConfig *config.SendConfig, generalConfig *config.GeneralConfig, log *zap.SugaredLogger) *Sender {
	sender := Sender{
		log:           log,
		queue:         queue,
		sendConfig:    sendConfig,
		generalConfig: generalConfig,
		chans:         make([]chan config.Message, len(sendConfig.Targets)),
	}
	return &sender

}

func (s *Sender) Run(ctx context.Context) {
	for idx, target := range s.sendConfig.Targets {

		capacity := s.generalConfig.ChannelCapacity.GetCapacity()
		s.log.Infof("Creating channel from Sender to goroutine with capacity %d", capacity)
		s.chans[idx] = make(chan config.Message, capacity)
		tr := &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: target.Insecure},
		}
		httpClient := &http.Client{Transport: tr}
		go s.ProcessNextMessage(ctx, httpClient, idx, target)
	}

	s.log.Debug("Sender will send to ", len(s.chans), " channels")
	for {
		select {
		case message := <-s.queue:

			s.BroadcastMessage(message)
		case <-ctx.Done():
			return
		}
	}
}

func (s *Sender) BroadcastMessage(message config.Message) {
	for idx, ch := range s.chans {

		log := s.log.With("request-id", message.RequestId)

		log.Debugf("Sender sending to channel %d (length=%d)", idx, len(ch))
		// verify there's room in our channel before bothering to publish these messages
		// note: if capacity==0, this means there is no capacity limit
		capacitySettings := s.generalConfig.ChannelCapacity
		if !capacitySettings.DropDisabled && len(ch) == capacitySettings.GetCapacity() {
			// no room, drop message
			log.Warnf("Sender dropping message to target %q as we've reached capacity %d", s.sendConfig.Targets[idx].Url, cap(ch))
			continue
		}

		ch <- message
	}
}

func (s *Sender) ProcessNextMessage(ctx context.Context, client *http.Client, index int, target config.Target) {

	log := s.log.With("target", target.Url)

	for {
		select {
		case message := <-s.chans[index]:
			requestLogger := log.With("request-id", message.RequestId)
			requestLogger.Debug(message)

			// sanity check to see if the message is too old
			if s.generalConfig.MaxAgeSeconds > 0 { //i.e. enabled
				maxAgeSeconds := s.generalConfig.MaxAgeSeconds

				requestLogger.Debugf("max age test: will compare %s to %s (original time=%s)", time.Now(), message.Timestamp.Add(time.Duration(maxAgeSeconds)*time.Second), message.Timestamp)
				if time.Now().After(message.Timestamp.Add(time.Duration(maxAgeSeconds) * time.Second)) {
					requestLogger.Warnf("Sender goroutine dropping message since current time is more than %d seconds from message time: %s", maxAgeSeconds, message.Timestamp)
					continue
				}

			}

			// using exponential backoff, send the message
			backoffStrategy := s.sendConfig.Backoff.Get()
			if backoffStrategy.MaxSteps < 1 {
				backoffStrategy.MaxSteps = 1
			}
			backoff := wait.Backoff{
				Steps:    backoffStrategy.MaxSteps,
				Duration: time.Duration(backoffStrategy.DurationSeconds) * time.Second,
				Factor:   backoffStrategy.Factor,
			}
			retryErr := wait.ExponentialBackoffWithContext(ctx, backoff, func(context.Context) (done bool, err error) {
				err = s.SendHTTPRequest(requestLogger, client, target, message)
				if err != nil {
					requestLogger.Debugf("HTTP Request failed: %v", err)
					return false, nil
				}
				return true, nil
			})
			if retryErr != nil {
				requestLogger.Errorf("HTTP Request failed: %v", retryErr)
			}

		case <-ctx.Done():
			return
		}
	}
}

// TODO: Implement sending HTTP request
func (s *Sender) SendHTTPRequest(log *zap.SugaredLogger, client *http.Client, target config.Target, message config.Message) error {
	log.Debugf("target: %+v, message: %q\n", target, message)
	//TODO make post request using s.client
	// 1. Send POST request to target. content type application/json
	// 2. handle response code
	// 3. retry 3 if response code 4xx and 5xx

	body := bytes.NewReader([]byte(message.Content))

	req, err := http.NewRequest("POST", target.Url, body)
	if err != nil {
		log.Errorf("Request error: %v", err)
		return err
	}

	res, err := client.Do(req)
	if err != nil {
		log.Errorf("Client error: %v", err)
		return err
	}

	if res != nil {
		if res.Body != nil {
			res.Body.Close()
		}
		log.Infof("Sender received Response - request-id=%q, status code: %d", message.RequestId, res.StatusCode)

		if res.StatusCode > 299 || res.StatusCode < 200 {
			return fmt.Errorf("bad status code: %d", res.StatusCode)
		}
	}

	if res == nil {
		return errors.New("response body empty")
	}

	return nil
}

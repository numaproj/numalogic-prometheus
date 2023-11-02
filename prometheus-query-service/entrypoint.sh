#!/bin/bash

## utils.sh gets placed here by the Dockerfile, and provides us with helpers to setup the environment for intuit
### https://github.intuit.com/data-curation/go/tree/master/intuit/utils.sh
source utils.sh
#
## IntuitEnsureJsonStd wraps any stdout/stderr lines that are non-JSON in json that matches OIL's slf format, including the output from this script.
### Because this repo uses the slf source-type for splunk it's important to ensure all logs are json so they do not get dropped.
IntuitEnsureJsonStd
#
## IntuitSSLCertGen generates the ssl certificates in use on the pod to provide a TLS endpoint for intuit gateway
IntuitSSLCertGen
#
## IntuitMeshSidecarWait will wait for mesh bootstraping if intuit mesh is enabled
IntuitMeshSidecarWait

exec ./main
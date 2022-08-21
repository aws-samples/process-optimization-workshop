#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Kill the primary component...
PID="`ps -ef | grep cstr_predictor.py | grep -v grep | awk '{print $2}'`"
if [ ! -z "${PID}" ]; then
   echo "Killing cstr_predictor.py PID:" ${PID} "..."
   kill -9 ${PID}
else
   echo "INFO: cstr_predictor.py does not appear to be running"
fi

# Exit
exit 0
#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

LOCATION="$1"
LOGFILE="/tmp/optimizer.log"

python3 -u ${LOCATION}/cstr_optimizer.py 1> ${LOGFILE} 2>&1
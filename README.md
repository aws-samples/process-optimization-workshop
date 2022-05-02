# Process Optimization Workshop

The AWS Process Optimization Workshop introduces participants to IoT and machine learning technologies that can extract insight from industrial data and use it to improve the performance of industrial processes. Participants will gain hands-on experience on using AWS IoT Greengrass to deploy and run software on edge devices. This workshop will illustrate how an edge device can be added to an industrial control network to publish sensor data to the cloud for near real-time monitoring and analysis of historical data. In the cloud, historical data will be used to build machine learning models for predictive (what-if analysis) and prescriptive (set point optimization) purposes. Finally, participants will learn how to deploy machine learning models to edge devices to leverage insights as part of latency-critical applications, such as set point optimization.

## Getting started

This repository contains the files and data required for the machine learning part of the AWS Process Optimization Workshop. If you launched the workshop from the AWS CloudFormation template, then you don't need to do anything else.

If you want to use this repository independently from the workshop, make sure to use a `ptorch_p36` kernel on Amazon SageMaker and to install the `dlr` library with the following script

```
pip install dlr
```

## Problem description

This workshop will simulate the preparation of a process model from historical data of an industrial process for use in process optimization. The process is a continuously stirred-tank reactor (CSTR), also known as backmix reactor, a common model for chemical reactors.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
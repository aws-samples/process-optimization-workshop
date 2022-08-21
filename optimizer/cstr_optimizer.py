# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from cgitb import handler
import grpc
import agent_pb2
import agent_pb2_grpc
import os
import json
import sys
import array
import traceback
import numpy as np
import time
import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT

## Additions for Optimizer
from scipy.optimize import dual_annealing
import pandas as pd
import timeit
import uuid

# Uncomment this for grpc debug log
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'

agent_socket = "unix:///tmp/aws.greengrass.SageMakerEdgeManager.sock"

agent_channel = grpc.insecure_channel(
    agent_socket, options=(("grpc.enable_http_proxy", 0),)
)

agent_client = agent_pb2_grpc.AgentStub(agent_channel)

main_loop_on = True
num_secs = 20
num_tries = 10
model_name = ""
result_topic = ""
handle = None
COMPONENT_CLIENTID = "CSTROptimizer"
model_loaded = True
model_unload = False
model_url = ""

def list_models():
    try:
        return agent_client.ListModels(agent_pb2.ListModelsRequest())
    except:
        e = sys.exc_info()[0]
        print("CSTR_Optimizer: Exception in list_models: " + str(e))
        print(traceback.format_exc())
        return


def list_model_tensors(models):
    return {
        model.name: {
            "inputs": model.input_tensor_metadatas,
            "outputs": model.output_tensor_metadatas,
        }
        for model in list_models().models
    }


def load_model(name, url):
    try:
        load_request = agent_pb2.LoadModelRequest()
        load_request.url = url
        load_request.name = name
        return agent_client.LoadModel(load_request)
    except:
        e = sys.exc_info()[0]
        print("CSTR_Optimizer: Exception in load_model: " + str(e))
        print(traceback.format_exc())
        return


def unload_model(name):
    try:
        unload_request = agent_pb2.UnLoadModelRequest()
        unload_request.name = name
        return agent_client.UnLoadModel(unload_request)
    except:
        e = sys.exc_info()[0]
        print("CSTR_Optimizer: Exception in unload_model: " + str(e))
        print(traceback.format_exc())
        return


def predict(model_name, input_data):
    global model_unload
    if model_unload == True:
        try:
            # print("CSTR_Optimizer: calling predict() on model: " + model_name + " with Input0: " + str(input_data))
            # Normalize per notebook
            X = (input_data["F"]-5.0)/95.0
            Y = (input_data["Q_dot"]+5000.0)/5000.0
            np_input_data = [X, Y]

            # Setup for predict() call...
            input_tensor = agent_pb2.Tensor()
            np_input_array = np.array(np_input_data).astype(np.float32)
            dimmed_np_input_array = np.expand_dims(np_input_array, axis=0)
            input_tensor.byte_data = dimmed_np_input_array.tobytes()
            input_tensor_metadata = list_model_tensors(list_models())[model_name]['inputs'][0]
            input_tensor.tensor_metadata.name = input_tensor_metadata.name
            input_tensor.tensor_metadata.data_type = input_tensor_metadata.data_type
            for shape in input_tensor_metadata.shape:
                input_tensor.tensor_metadata.shape.append(shape)
            predict_request = agent_pb2.PredictRequest()
            predict_request.name = model_name
            predict_request.tensors.append(input_tensor)
            predict_response = agent_client.Predict(predict_request)
            # print("CSTR_Optimizer: predict() raw Response: ")
            # print(predict_response)
            for tensor in predict_response.tensors:
                # We have an output tensor
                output_array = array.array('f', tensor.byte_data)
                output_data = output_array.tolist()
                fmt_output_data = [ "%.4f" % float(elem) for elem in output_data ]
                # print("CSTR_Optimizer: predict() numeric result - Input0: " + str(dimmed_np_input_array) + " --> Output: " + str(fmt_output_data))

            # DA: Create compatible format response
            fmt_predict_response = {}
            fmt_predict_response['F'] = round(input_data["F"],4)
            fmt_predict_response['Q_dot'] = round(input_data["Q_dot"],4)
            fmt_predict_response['C_a'] = round(float(fmt_output_data[0]),4)
            fmt_predict_response['C_b'] = round(float(fmt_output_data[1]),4)
            fmt_predict_response['T_K'] = round((float(fmt_output_data[2])*25.0)+125.0,1)
            return fmt_predict_response
        except:
            e = sys.exc_info()[0]
            print("CSTR_Optimizer: Exception in predict: " + str(e))
            print(traceback.format_exc())
            fmt_predict_response = {}
            fmt_predict_response['F'] = round(input_data["F"],4)
            fmt_predict_response['Q_dot'] = round(input_data["Q_dot"],4)
            return fmt_predict_response
    else:
        print("CSTR_Optimizer: No models loaded")
        fmt_predict_response = {}
        fmt_predict_response['F'] = round(input_data["F"],4)
        fmt_predict_response['Q_dot'] = round(input_data["Q_dot"],4)
        return fmt_predict_response

def connect_iotcore(gg_install_root,gg_endpoint_uuid,root_cert_filename,gg_endpoint_url):
    # Custom connection info for our GG device
    ENDPOINT = gg_endpoint_url

    # if we are running GGV2 via CloudFormation, we adjust these
    if gg_endpoint_uuid == "cloudformation":
        # With Cloud Formation deployed GGv2, these filenames are used
        PATH_TO_CERTIFICATE = gg_install_root + "/" + "thingCert.crt"
        PATH_TO_PRIVATE_KEY = gg_install_root + "/" + "privKey.key"
        PATH_TO_AMAZON_ROOT_CA_1 = gg_install_root  + "/" + root_cert_filename
    else:
        # Manual GGv2 deployment
        PATH_TO_CERTIFICATE = gg_install_root + "/certificates/" + gg_endpoint_uuid + "-cert.pem"
        PATH_TO_PRIVATE_KEY = gg_install_root + "/certificates/" + gg_endpoint_uuid + "-privkey.pem"
        PATH_TO_AMAZON_ROOT_CA_1 = gg_install_root + "/certificates/" + root_cert_filename

    myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient(COMPONENT_CLIENTID)
    myAWSIoTMQTTClient.configureEndpoint(ENDPOINT, 8883)
    myAWSIoTMQTTClient.configureCredentials(PATH_TO_AMAZON_ROOT_CA_1, PATH_TO_PRIVATE_KEY, PATH_TO_CERTIFICATE)
    myAWSIoTMQTTClient.connect()
    return myAWSIoTMQTTClient

def publish_results_to_iotcore(handle,input0,result,result_topic):
    try:
        message = {"input":input0, "result":result}
        handle.publishAsync(result_topic, json.dumps(message), 1, None)
    except:
        e = sys.exc_info()[0]
        print("CSTR_Optimizer: Exception in publish_results_to_iotcore: " + str(e))

    return

def stop_received(json_input):
    try:
        check = json_input['shutdown']
        if "yes" in check:
            return True
        else:
            return False
    except:
        return False

#### Optimizer: 
def optimal_manipulated_vars(model_name, C_b_ref, max_iterations):
    """
    Drive CSTR toward C_b_ref
    """
    C_b_ref = C_b_ref
    # Admissible bounds for manipulated variables
    bounds = ((5.0, 100.0), (-5000.0, 0.0))
    
    def objective(x):
        '''
        Objective function to minimize: Penalizes deviations from
        C_b_ref and T_K leaving the desired range
        '''
        # DA: COMMENT OUT: manipulated_vars = pd.DataFrame({"F": [x[0]], "Q_dot" : [x[1]]})
        # DA: Create compatiable {F, Q_dot} input for the predict() function...
        predict_input = {"F":x[0], "Q_dot":x[1]}

        # DA: call predict()...
        prediction = predict(model_name, predict_input)

        cost = 0.0
        # Deviation from reference C_b_ref
        cost += (prediction["C_b"]-C_b_ref)**2
        # Cooling jacket temperature
        if (prediction["T_K"] < 5.0):
            cost += 0.01*(prediction["T_K"]-5.0)**2
        elif (prediction["T_K"] > 140.0):
            cost += 0.01*(prediction["T_K"]-140.0)**2

        return cost

    # DA: Cleaning up output for nice logging...
    print("CSTR_Optimizer: Starting dual_annealing() iterations (Max: " + str(max_iterations) + " iterative predict() calls)...")
    tic=timeit.default_timer()
    result = dual_annealing(objective, bounds=bounds, maxiter=max_iterations)
    toc=timeit.default_timer()
    print("CSTR_Optimizer: Completed dual_annealing().")
    
    return (result['x'], result['nfev'], toc-tic)
#### Optimizer

def process_input(client, userdata, message):
    global main_loop_on
    global handle
    global model_name
    global model_unload
    global model_url

    # make sure our model is loaded
    if model_unload == False:
        print("CSTR_Optimizer: process_input(): Loading model: " + model_url)
        bind_to_seam(model_url)

    json_input = json.loads(message.payload.decode())
    if stop_received(json_input):
        main_loop_on = False
    else:
        if model_unload == True:
            # Make sure we have valid params
            if "C_b_ref" in json_input:
                # Assign a UUID if one was not provided
                my_uuid = str(uuid.uuid4())
                if "uuid" in json_input:
                    my_uuid = json_input['uuid']

                # Immediately send dispatch that optimizer invoked
                in_progress = {"status": "running", "uuid": my_uuid}
                print("CSTR_Optimizer: STARTING: Optimization started: " + str(in_progress) + " to topic: " + result_topic + "...")
                publish_results_to_iotcore(handle,json_input,in_progress,result_topic)

                # Invoke the optimizer... LONG WINDED...
                print("CSTR_Optimizer: Calling optimal_manipulated_vars() with C_b_ref: " + str(json_input['C_b_ref']) + " Max Iterations: " + str(json_input['max_iterations']))
                x, nfev, elapsed_secs = optimal_manipulated_vars(model_name, json_input['C_b_ref'], json_input['max_iterations'])

                # Call predict once more on the final optimization values
                predict_input = {"F":x[0], "Q_dot":x[1]}

                # DA: call a final predict() per the notebook...
                print("predict(): Input Args: " + str(predict_input) + " Model Name: " + model_name)
                prediction = predict(model_name, predict_input)

                # DA: Format the result to match evaluate_pytorch_model()
                prediction['C_b_ref'] = json_input['C_b_ref']
                prediction['iterations'] = nfev
                prediction['status'] = "finished"
                prediction['uuid'] = my_uuid
                prediction['execute_time_secs'] = round(elapsed_secs,1)

                # post the results to IoTCore MQTT topic
                print("CSTR_Optimizer: FINISHED: Optimizing results: " + str(prediction) + " to topic: " + result_topic + "...")
                publish_results_to_iotcore(handle,json_input,prediction,result_topic)
            else:
                # invalid params... ignore
                print("CSTR_Optimizer: Invalid parameters supplied. Please check input JSON...must contain C_b_ref key.")
                fmt_predict_response = {"status":"error","info":"missing C_b_ref key in input"}
                publish_results_to_iotcore(handle,json_input,fmt_predict_response,result_topic)
        else:
            # invalid params... ignore
            print("CSTR_Optimizer: No models appear loaded.")
            fmt_predict_response = {"status":"error","info":"no models appear loaded"}
            publish_results_to_iotcore(handle,json_input,fmt_predict_response,result_topic)


def main_loop(model_name):
    print("CSTR_Optimizer: Entering main loop()...")
    global main_loop_on
    while main_loop_on == True:
        time.sleep(num_secs)

def bind_to_seam(model_url):
    global model_loaded
    global model_unload

    # See if our model is already loaded
    try: 
        list = list_models()
        for model in list.models:
            print("CSTR_Optimizer: Found Model: " + model.name)
            if model.name == model_name:
                print("CSTR_Optimizer: Model: " + model_name + " already loaded. OK")
                model_loaded = False
                model_unload = True
                break
            else:
                print("CSTR_Optimizer: Model: " + model_name + " NOT loaded. Loading...")
        else:
            print("CSTR_Optimizer: No models loaded. Loading: " + model_name + "...")
    except:
        print("CSTR_Optimizer: No models loaded. Loading: " + model_name + "...")

    # Load the model
    if model_loaded == True: 
        print("CSTR_Optimizer: Loading model: " + model_name + " URL: " + model_url)
        load_model(model_name, model_url)
        model_unload = True

    # List the model
    print("CSTR_Optimizer: Listing Models:")
    print(list_models())
    list = list_models()
    if hasattr(list, 'models'):
        if len(list.models) > 0:
            print("First Model: " + list.models[0].name)
        else:
            print("No models loaded (OK).")
    else:
        print("No models loaded (grpc down).")

def main():
    global main_loop_on
    global model_name
    global result_topic
    global handle
    global model_url
    global num_tries
    global model_unload

    # Defaults
    parameters = json.loads(os.getenv("AWS_CSTR_PARAMETERS"))
    model_name = parameters['model_name']
    target_device = parameters['target_device']
    gg_install_root = parameters['gg_install_root']
    gg_endpoint_url = parameters['gg_endpoint_url']
    gg_endpoint_uuid = parameters['gg_endpoint_uuid']
    root_cert_filename = parameters['root_cert_filename']
    result_topic = parameters['result_topic']
    input_topic = parameters['input_topic']
    model_url = gg_install_root + "/work/" + model_name + "-" + target_device + "-component"

    try:
        # Display the configuration
        print("CSTR_Optimizer: Configuration: Model: " + model_name + " URL: " + model_url)
        print("CSTR_Optimizer: Socket: " + agent_socket)

        # Bind to SageMaker Edge Agent Manager (seam)
        i = 0
        while model_unload == False and i < num_tries:
            bind_to_seam(model_url)
            time.sleep(num_secs)
            i = i + 1

        # Connect to IoTCore for results publication to MQTT
        print("CSTR_Optimizer: Connecting to IoTCore for MQTT results publishing...")
        handle = connect_iotcore(gg_install_root,gg_endpoint_uuid,root_cert_filename,gg_endpoint_url)

        # Subscribe to the input topic
        print("CSTR_Optimizer: Subscribing to input topic: " + input_topic)
        handle.subscribe(input_topic, 1, process_input)

        # Enter the main loop looking for inputs and calling predict()
        main_loop(model_name)

        # unload the model as the event loop is exiting...
        if model_unload == True:
            print("CSTR_Optimizer: Unloading model: " + model_name)
            unload_model(model_name)

        # Close down the MQTT connection
        print("CSTR_Optimizer: Main loop closing... disconnecting from AWS...")
        handle.disconnect()
    except:
        e = sys.exc_info()[0]
        print("CSTR_Optimizer: Exception in main: " + str(e))
        print(traceback.format_exc())
        
    # DONE
    print("CSTR_Optimizer: DONE! Exiting.")

if __name__ == "__main__":
    main()

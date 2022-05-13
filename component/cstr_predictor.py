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

# Uncomment this for grpc debug log
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'

agent_socket = "unix:///tmp/aws.greengrass.SageMakerEdgeManager.sock"

agent_channel = grpc.insecure_channel(
    agent_socket, options=(("grpc.enable_http_proxy", 0),)
)

agent_client = agent_pb2_grpc.AgentStub(agent_channel)

main_loop_on = True
num_secs = 10
model_name = ""
result_topic = ""
handle = None

def list_models():
    try:
        return agent_client.ListModels(agent_pb2.ListModelsRequest())
    except:
        e = sys.exc_info()[0]
        print("CSTR_Predictor: Exception in list_models: " + str(e))
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
        print("CSTR_Predictor: Exception in load_model: " + str(e))
        print(traceback.format_exc())
        return


def unload_model(name):
    try:
        unload_request = agent_pb2.UnLoadModelRequest()
        unload_request.name = name
        return agent_client.UnLoadModel(unload_request)
    except:
        e = sys.exc_info()[0]
        print("CSTR_Predictor: Exception in unload_model: " + str(e))
        print(traceback.format_exc())
        return


def predict(model_name, input_data):
    try:
        print("CSTR_Predictor: calling predict() on model: " + model_name + " with Input0: " + str(input_data))
        input_tensor = agent_pb2.Tensor()
        np_input_array = np.array(input_data).astype(np.float32)
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
        print("CSTR_Predictor: predict() raw Response: ")
        print(predict_response)
        for tensor in predict_response.tensors:
            # We have an output tensor
            output_array = array.array('f', tensor.byte_data)
            output_data = output_array.tolist()
            fmt_output_data = [ '%.4f' % elem for elem in output_data ]
            print("CSTR_Predictor: predict() numeric result - Input0: " + str(dimmed_np_input_array) + " --> Output: " + str(fmt_output_data))
        return fmt_output_data
    except:
        e = sys.exc_info()[0]
        print("CSTR_Predictor: Exception in predict: " + str(e))
        print(traceback.format_exc())
        return []

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
        PATH_TO_CERTIFICATE = gg_install_root + "/certificates/" + gg_endpoint_uuid + "-certificate.pem.crt"
        PATH_TO_PRIVATE_KEY = gg_install_root + "/certificates/" + gg_endpoint_uuid + "-private.pem.key"
        PATH_TO_AMAZON_ROOT_CA_1 = gg_install_root + "/certificates/" + root_cert_filename

    myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient("")
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
        print("CSTR_Predictor: Exception in publish_results_to_iotcore: " + str(e))

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

def process_input(client, userdata, message):
    global main_loop_on
    global handle
    global model_name

    json_input = json.loads(message.payload.decode())
    if stop_received(json_input):
        main_loop_on = False
    else:
        # Invoke the prediction function
        print("CSTR_Predictor: Calling predict() with Input: " + str(json_input['input']))
        result = predict(model_name, json_input['input'])

        # post the results to IoTCore MQTT topic
        print("CSTR_Predictor: Publishing results: " + str(result) + " to topic: " + result_topic + "...")
        publish_results_to_iotcore(handle,json_input['input'],result,result_topic)

def main_loop(model_name):
    print("CSTR_Predictor: Entering main loop()...")
    global main_loop_on
    while main_loop_on == True:
        time.sleep(num_secs)

def main():
    global main_loop_on
    global model_name
    global result_topic
    global handle

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
    model_loaded = True
    model_unload = False

    try:
        # Display the configuration
        print("CSTR_Predictor: Configuration: Model: " + model_name + " URL: " + model_url)
        print("CSTR_Predictor: Socket: " + agent_socket)

        # Connect to IoTCore for results publication to MQTT
        print("CSTR_Predictor: Connecting to IoTCore for MQTT results publishing...")
        handle = connect_iotcore(gg_install_root,gg_endpoint_uuid,root_cert_filename,gg_endpoint_url)

        # See if our model is already loaded
        list = list_models()
        for model in list.models:
            print("CSTR_PredictorFirst Model: " + model.name)
            if model.name == model_name:
                print("CSTR_Predictor: Model: " + model_name + " already loaded. OK")
                model_loaded = False
                model_unload = True
                break
            else:
                print("CSTR_Predictor: Model: " + model_name + " NOT loaded. Loading...")
        else:
            print("CSTR_Predictor: No models loaded. Loading: " + model_name + "...")

        # Load the model
        if model_loaded == True: 
            print("CSTR_Predictor: Loading model: " + model_name + " URL: " + model_url)
            load_model(model_name, model_url)
            model_unload = True

        # List the model
        print("CSTR_Predictor: Listing Models:")
        print(list_models())
        list = list_models()
        if hasattr(list, 'models'):
            if len(list.models) > 0:
                print("First Model: " + list.models[0].name)
            else:
                print("No models loaded (OK).")
        else:
            print("No models loaded (grpc down).")

        # Subscribe to the input topic
        print("CSTR_Predictor: Subscribing to input topic: " + input_topic)
        handle.subscribe(input_topic, 1, process_input)

        # Enter the main loop looking for inputs and calling predict()
        main_loop(model_name)

        # unload the model as the event loop is exiting...
        if model_unload == True:
            print("CSTR_Predictor: Unloading model: " + model_name)
            unload_model(model_name)

        # Close down the MQTT connection
        print("CSTR_Predictor: Main loop closing... disconnecting from AWS...")
        handle.disconnect()
    except:
        e = sys.exc_info()[0]
        print("CSTR_Predictor: Exception in main: " + str(e))
        print(traceback.format_exc())
        
    # DONE
    print("CSTR_Predictor: DONE! Exiting.")

if __name__ == "__main__":
    main()

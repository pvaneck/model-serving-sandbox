## gRPC Inference using KFServing V2 Protocol

The gRPC files were generated from the KFServing [grpc_predict_v2.proto](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/grpc_predict_v2.proto) file.

Command used to generate:

```sh
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc_predict_v2.proto
```

The sample client code expects the serving service to be port-forwarded:

Example:

```
kubectl port-forward --address 0.0.0.0 service/<service name> 8033 -n <namespace>
```


### Usage:

Install dependencies in virtualenv.
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run inference.

```sh
python client.py
```

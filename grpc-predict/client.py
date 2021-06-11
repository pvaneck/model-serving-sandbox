import random

import grpc_predict_v2_pb2 as pb
import grpc_predict_v2_pb2_grpc as pb_grpc
import grpc

from sklearn import datasets


if __name__ == "__main__":
    channel = grpc.insecure_channel('localhost:8033')
    infer_client = pb_grpc.GRPCInferenceServiceStub(channel)

    # Load sample digit image from sklearn datasets.
    digits = datasets.load_digits()
    num_digits = len(digits.images)
    rand_int = random.randint(0, num_digits-1)
    random_digit_image = digits.images[rand_int]
    random_digit_label = digits.target[rand_int]
    print(random_digit_image)
    data = random_digit_image.reshape((1, -1))
    print(data)

    tensor_contents = pb.InferTensorContents(fp32_contents=data[0])

    infer_input = pb.ModelInferRequest().InferInputTensor(
        name="input-0",
        shape=[1,64],
        datatype="FP32",
        contents=tensor_contents
    )

    # metadata = (('mm-vmodel-id','example-sklearn-mnist-svm'),)
    inputs = [infer_input]
    request = pb.ModelInferRequest(model_name="example-sklearn-mnist-svm", inputs=inputs)

    results, call = infer_client.ModelInfer.with_call(request=request)
    print(results)
    print('Expected Result: {}'.format(random_digit_label))

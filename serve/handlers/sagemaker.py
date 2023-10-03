import os
import io
import json
import boto3
from handlers.base import BaseModel

class model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.endpoint_name = model_name
        self.sagemaker_client = boto3.client(
            service_name="sagemaker-runtime"
        )
        self.invoke_api = self.sagemaker_client.invoke_endpoint_with_response_stream

    def invoke(self, body):
        response = self.invoke_api(
            EndpointName=self.endpoint_name,
            Body = json.dumps(body).encode("utf-8"),
            ContentType="application/json"
        )["Body"]
        return response
    
class StreamIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == 10:
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'PayloadPart' not in chunk:
                print(f"Unknown event type: {chunk}")
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])
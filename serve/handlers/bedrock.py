import os
import io
import json
import boto3
from handlers.base import BaseModel


class model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime", 
            region_name="us-west-2"
        )
        self.invoke_api = self.bedrock_client.invoke_model_with_response_stream

    def invoke(self, body):
        try:
            response = self.invoke_api(
                modelId=self.model_name,
                body = json.dumps(body).encode("utf-8")
            )["body"]
            return response
        except Exception as e:
            print(f"Error {e}, Body {body}")
            raise e
            
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
            if line:
                self.read_pos += len(line)
                return line
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'chunk' not in chunk:
                print(f"Unknown event type: {chunk}")
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['chunk']['bytes'])
        

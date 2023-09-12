import os
import requests
import datetime
import time
import json
import gradio as gr
import boto3

from fastchat.serve.gradio_web_server import (
    logger,
    enable_btn,
    disable_btn,
    no_change_btn,
    model_worker_stream_iter,
    post_process_code
)
from fastchat.constants import (
    SERVER_ERROR_MSG,
    ErrorCode
)
from fastchat.serve.api_provider import (
    anthropic_api_stream_iter,
    openai_api_stream_iter,
    palm_api_stream_iter
)

worker_addr = "http://127.0.0.1:21002"

class conv_logger:
    def __init__(self):
        self.log_bucket = ""
        self.log_prefix = ""


def upload_to_s3(bucket, prefix, filename, data):
    s3_client.put_object(
        Body=data, 
        Bucket=bucket, 
        Key=os.path.join(prefix, filename)
    )


def get_conv_log_filename(conv_id, log_type):
    t = datetime.datetime.utcnow()
    name = f"conv-{t.hour:02d}-{t.minute:02d}-{t.second:02d}.json"
    prefix = f"{log_type}/{t.year}/{t.month:02d}/{t.day:02d}/{conv_id}"
    return os.path.join(prefix, name)


def bot_response(state, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"bot_response. ip: {request.client.host}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    conv, model_name = state.conv, state.model_name
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name == "claude-2" or model_name == "claude-instant-1":
        prompt = conv.get_prompt()
        stream_iter = anthropic_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name == "palm-2":
        stream_iter = palm_api_stream_iter(
            state.palm_chat, conv.messages[-2][1], temperature, top_p, max_new_tokens
        )
    else:
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
            )
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()

        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
        )

    conv.update_last_message("▌")
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        for data in stream_iter:
            if data["error_code"] == 0:
                output = data["text"].strip()
                if "vicuna" in model_name:
                    output = post_process_code(output)
                conv.update_last_message(output + "▌")
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
            time.sleep(0.015)
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Delete "▌"
    conv.update_last_message(conv.messages[-1][-1][:-1])
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    filename = get_conv_log_filename(state.conv_id, "conv")
    data = {
        "tstamp": round(finish_tstamp, 4),
        "type": "chat",
        "model": model_name,
        "gen_params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "state": state.dict(),
        "ip": request.client.host,
    }
    upload_to_s3(
        conv_log.log_bucket,
        conv_log.log_prefix,
        filename,
        json.dumps(data)
        )


s3_client = boto3.client("s3")
conv_log = conv_logger()
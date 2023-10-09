"""
The gradio webui server for chatting with a single model.
"""
import argparse
import time
import json
from urllib.parse import urlparse
import gradio as gr
import boto3
from fastchat.utils import (
    get_window_url_params_js,
    parse_gradio_auth_creds,
    violates_moderation
)
from fastchat.serve.gradio_web_server import (
    logger,
    enable_moderation,
    block_css,
    flag_last_response,
    ip_expiration_dict,
    disable_btn,
    no_change_btn,
)
from fastchat.serve.gradio_block_arena_named import (
    set_global_vars_named,
)
from fastchat.constants import (
    SESSION_EXPIRATION_TIME,
    CONVERSATION_LIMIT_MSG,
    CONVERSATION_TURN_LIMIT,
    INACTIVE_MSG,
    MODERATION_MSG,
    INPUT_CHAR_LEN_LIMIT,
)
from fastchat.model.model_adapter import (
    BaseModelAdapter,
    model_adapters
)
from fastchat.conversation import Conversation, get_conv_template
import fastchat
from gradio_block_arena_named import (
    build_side_by_side_ui_named,
    load_webui_side_by_side_named,
    SageMakerState
)
from gradio_utils import (
    bot_response,
    upload_to_s3,
    get_conv_log_filename,
    conv_log,
    worker_addr
)
import gradio_utils as utils
from io import StringIO

s3_client = boto3.client("s3")
ddb_client = boto3.client("dynamodb")
conv_templates = ""


def set_global_vars(ddb_table_name_, enable_moderation_, conv_templates_):
    global ddb_table_name, enable_moderation, conv_templates
    ddb_table_name = ddb_table_name_
    enable_moderation = enable_moderation_
    conv_templates = conv_templates_


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def load_webui_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.UploadButton.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def add_text(state, model_selector, text, request: gr.Request):
    ip = request.client.host
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = SageMakerState(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5

    if ip_expiration_dict[ip] < time.time():
        logger.info(f"inactive. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), INACTIVE_MSG) + (no_change_btn,) * 5

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), MODERATION_MSG) + (
                no_change_btn,
            ) * 5

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def load_webui(url_params, request: gr.Request):
    global models

    ip = request.client.host
    logger.info(f"load_webui. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    if args.model_list_mode == "reload":
        models = get_model_list(
            ddb_table_name
        )

    return load_webui_single(models, url_params)


def send_file(state, model_selector, file_box, request: gr.Request):
    logger.info(f"send_file. ip: {request.client.host}")
    if state is None:
        state = SageMakerState(model_selector)

    if file_box is None:
        return (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5

    conv = state.conv

    if file_box.name.lower().endswith((".txt", ".csv", ".sh", ".py", ",java")):
        stringio = StringIO(open(file_box.name).read())
        content = "=== BEGIN FILE ===\n"
        content += stringio.read().strip()
        content += "\n=== END FILE ===\n\n Instruction: Please confirm that you have read that file by saying: 'Yes, I have read the file'"

        conv.append_message(conv.roles[0], content, "I have uploaded a file. Please confirm that you have read that file.")
        conv.append_message(conv.roles[1], None)
    else:
        conv.append_message(conv.roles[0], "Uploaded file")
        conv.append_message(conv.roles[1], None)

    return (state, state.to_gradio_chatbot()) + (disable_btn,) * 5


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.conv.update_last_message(None)
    load_templates_from_s3(conv_templates)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    load_templates_from_s3(conv_templates)
    ip = request.client.host
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME
    return (state, [], "") + (disable_btn,) * 5


def build_single_model_ui(models):

    notice_markdown = f"""
# 🏔️ Chat with Open Large Language Models

### Choose a model to chat with
"""

    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
            container=False,
        )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="Scroll down and start chatting",
        visible=False,
        height=550,
    )
    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
                container=False,
            )
        with gr.Column(scale=1, min_width=50):
            send_file_btn = gr.UploadButton(label="📂️", visible=False, size="lg", interactive=True, type="file", file_types=["text"])

    with gr.Row(visible=False) as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=8196,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    textbox.submit(
        add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_file_btn.upload(
        send_file, [state, model_selector, send_file_btn], [state, chatbot] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return (
        state, 
        model_selector, 
        chatbot,
        textbox, 
        send_file_btn, 
        button_row, 
        parameter_row
        )


def build_webui_single(models):
    with gr.Blocks(
        title="Chat with Open Large Language Models hosted on SageMaker",
        theme=gr.themes.Base(),
        css=block_css,
    ) as webui:
        url_params = gr.JSON(visible=False)

        (
            state,
            model_selector,
            chatbot,
            textbox, 
            send_file_btn,
            button_row,
            parameter_row
        ) = build_single_model_ui(models)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")
        webui.load(
            load_webui,
            [url_params],
            [
                state,
                model_selector,
                chatbot,
                textbox,
                send_file_btn,
                button_row,
                parameter_row
            ],
            _js=get_window_url_params_js,
        )

    return webui


def load_webui_multi(url_params, request: gr.Request):
    global models

    ip = request.client.host
    logger.info(f"load_webui_multi. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    selected = 0
    if "compare" in url_params:
        selected = 1

    models = get_model_list(
        ddb_table_name
    )

    single_updates = load_webui_single(models, url_params)

    side_by_side_named_updates = load_webui_side_by_side_named(models, url_params)
    return (
        (gr.Tabs.update(selected=selected),)
        + single_updates
        + side_by_side_named_updates
    )


def build_webui_multi(models):
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Base(),
        css=block_css,
    ) as webui:
        with gr.Tabs() as tabs:
            with gr.Tab("Single Model", id=0):
                (
                    a_state,
                    a_model_selector,
                    a_chatbot,
                    a_textbox,
                    a_send_btn,
                    a_button_row,
                    a_parameter_row,
                ) = build_single_model_ui(models)
                a_list = [
                    a_state,
                    a_model_selector,
                    a_chatbot,
                    a_textbox,
                    a_send_btn,
                    a_button_row,
                    a_parameter_row,
                ]

            with gr.Tab("Chatbot Arena (side-by-side)", id=1):
                (
                    b_states,
                    b_model_selectors,
                    b_chatbots,
                    b_textbox,
                    b_send_btn,
                    b_button_row,
                    b_button_row2,
                    b_parameter_row,
                ) = build_side_by_side_ui_named(models)
                b_list = (
                    b_states
                    + b_model_selectors
                    + b_chatbots
                    + [
                        b_textbox,
                        b_send_btn,
                        b_button_row,
                        b_button_row2,
                        b_parameter_row,
                    ]
                )

        url_params = gr.JSON(visible=False)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")
        webui.load(
            load_webui_multi,
            [url_params],
            [tabs] + a_list + b_list,
            _js=get_window_url_params_js,
        )

    return webui


def load_templates_from_s3(url):
    if url:
        scheme, netloc, path, _, _, _ = urlparse(url)

        if scheme == "s3":
            path = path[1:] if len(path) > 0 and path.startswith("/") else path
            if path.endswith(".json"):
                if netloc:
                    logger.info(f"load_templates: {url}")
                    result = s3_client.get_object(Bucket=netloc, Key=path)
                    json_templates = json.loads(
                        result["Body"].read().decode(),
                        object_hook=lambda d: Conversation(**d) if not isinstance(list(d.values())[0], Conversation) else d, 
                        strict=False
                        )
                    for conv_template, template in json_templates.items():
                        fastchat.conversation.conv_templates[conv_template] = template


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    filename = get_conv_log_filename(state.conv_id, "vote")
    data = {
        "tstamp": round(time.time(), 4),
        "type": vote_type,
        "model": model_selector,
        "state": state.dict(),
        "ip": request.client.host,
    }
    upload_to_s3(
        conv_log.log_bucket,
        conv_log.log_prefix,
        filename,
        json.dumps(data)
        )


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3

def get_model_list(ddb_table_name):
    ddb_client = boto3.client("dynamodb")
    return ddb_client.get_item(
        TableName=ddb_table_name, 
        Key={
            "model_name": {
                "S": "all-models"
            }
        }
    )["Item"]["model_names"]["S"].split(",")

class ClaudeAdapter(BaseModelAdapter):
    """The model adapter for Claude"""

    def match(self, model_path: str):
        return model_path in ["anthropic.claude-v2", "anthropic.claude-instant-v1", "anthropic.claude-v1"]

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("claude")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue.",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="WebUI mode, single model or side by side multi",
    )
    parser.add_argument(
        "--moderate", action="store_true", help="Enable content moderation"
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
        default=None,
    )
    parser.add_argument(
        "--conv-templates", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-bucket", type=str, default=None, help="S3 Bucket to store conversation logs"
    )
    parser.add_argument(
        "--conv-prefix", type=str, default=None, help="S3 prefix to store conversation"
    )
    parser.add_argument(
        "--ddb-table-name", type=str, default=None, help="DDB table for list of models"
    )
    parser.add_argument(
        "--worker-address", type=str, default="http://127.0.0.1:21002", help="Model worker address"
    )
    args = parser.parse_args()
    fastchat.serve.gradio_web_server.args = args
    logger.info(f"args: {args}")
    worker_addr = args.worker_address

    load_templates_from_s3(args.conv_templates)
    fastchat.model.model_adapter.model_adapters.append(ClaudeAdapter())

    # Set global variables
    set_global_vars(
        args.ddb_table_name,
        args.moderate,
        args.conv_templates
        )
    set_global_vars_named(args.moderate)
    conv_log.log_bucket = args.conv_bucket
    conv_log.log_prefix = args.conv_prefix
    fastchat.serve.gradio_block_arena_named.set_global_vars_named(args.moderate)
    models = get_model_list(
        args.ddb_table_name
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the webui
    if args.mode == "single":
        webui = build_webui_single(models)
    else:
        # Multi agent UI
        webui = build_webui_multi(models)
    webui.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
    )
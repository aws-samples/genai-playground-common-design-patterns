"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import dataclasses
import time
import json
from typing import List
import gradio as gr
import numpy as np
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INACTIVE_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.serve.gradio_web_server import (
    no_change_btn,
    disable_btn,
    get_model_description_md,
    ip_expiration_dict,
    State
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
)

from fastchat.serve.gradio_block_arena_named import (
    regenerate,
    flash_buttons,
    clear_history
)
from fastchat.conversation import Conversation
from gradio_utils import (
    bot_response,
    upload_to_s3,
    get_conv_log_filename,
    conv_log
)


class SageMakerState(State):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.conv.__class__ = SageMakerConversation
        self.conv.visible_messages = self.conv.messages.copy()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )
        return base


@dataclasses.dataclass
class SageMakerConversation(Conversation):

    visible_messages: List[List[str]] = ()

    def append_message(self, role: str, message: str, visible_message: str = None):
        super().append_message(role, message)
        if visible_message:
            self.visible_messages.append([role, visible_message])
        else:
            self.visible_messages.append([role, message])

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.visible_messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        super().update_last_message(message)
        self.visible_messages[-1][1] = message

    def copy(self):
        return SageMakerConversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            visible_messages=[[x, y] for x, y in self.visible_messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        ret = super().dict()
        ret.update({"visible_messages": self.visible_messages})
        return ret


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    filename = get_conv_log_filename(states[0].conv_id, "vote")
    data = {
        "tstamp": round(time.time(), 4),
        "type": vote_type,
        "models": [x for x in model_selectors],
        "states": [x.dict() for x in states],
        "ip": request.client.host,
    }
    upload_to_s3(
        conv_log.log_bucket,
        conv_log.log_prefix,
        filename,
        json.dumps(data)
        )


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def load_webui_side_by_side_named(models, url_params):
    states = (None,) * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 32)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = (
        gr.Dropdown.update(choices=models, value=model_left, visible=True),
        gr.Dropdown.update(choices=models, value=model_right, visible=True),
    )

    return (
        states
        + selector_updates
        + (gr.Chatbot.update(visible=True),) * num_sides
        + (
            gr.Textbox.update(visible=True),
            gr.Box.update(visible=True),
            gr.Row.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True),
        )
    )


def add_text(
    state0, state1, model_selector0, model_selector1, text, request: gr.Request
):
    ip = request.client.host
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = SageMakerState(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                no_change_btn,
            ]
            * 6
        )

    if ip_expiration_dict[ip] < time.time():
        logger.info(f"inactive (named). ip: {request.client.host}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [INACTIVE_MSG]
            + [
                no_change_btn,
            ]
            * 6
        )

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(
                f"violate moderation (named). ip: {request.client.host}. text: {text}"
            )
            for i in range(num_sides):
                states[i].skip_next = True
            return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [MODERATION_MSG]
                + [
                    no_change_btn,
                ]
                * 6
            )

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {request.client.host}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 6
    )


def send_file(state0, state1, model_selector0, model_selector1, file_box, request: gr.Request):
    logger.info(f"send_file. ip: {request.client.host}")
    states = [
        state0 if state0 is not None else SageMakerState(model_selector0),
        state1 if state1 is not None else SageMakerState(model_selector1)
        ]
    for state in states:
        if file_box is None:
            return states + [x.to_gradio_chatbot() for x in states] + [no_change_btn] * 6
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
    return states + [x.to_gradio_chatbot() for x in states] + [disable_btn] * 6


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    def collector_func(gen, future):
        gen.append(future.result())

    logger.info(f"bot_response_multi (named). ip: {request.client.host}")

    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    with ThreadPoolExecutor() as executor:

        for i in range(num_sides):
            kwargs = {
                "state": states[i],
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "request": request
            }
            future = executor.submit(bot_response, **kwargs)
            future.add_done_callback(partial(collector_func, gen))

    chatbots = [None] * num_sides
    while True:
        stop = True
        for i in range(num_sides):
            try:
                ret = next(gen[i])
                states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break


def build_side_by_side_ui_named(models):
    notice_markdown = """
# ⚔️  Chatbot Arena ⚔️ 
### Rules
- Chat with two models side-by-side and vote for which one is better!
- You pick the models you want to chat with.
- You can do multiple rounds of conversations before voting.
- Click "Clear history" to start a new round.

### Choose two models to chat with
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    with gr.Box(elem_id="share-region-named"):
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )

        with gr.Row():
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label, elem_id="chatbot", visible=False, height=550
                    )

        with gr.Box() as button_row:
            with gr.Row():
                leftvote_btn = gr.Button(value="👈  A is better", interactive=False)
                rightvote_btn = gr.Button(value="👉  B is better", interactive=False)
                tie_btn = gr.Button(value="🤝  Tie", interactive=False)
                bothbad_btn = gr.Button(value="👎  Both are bad", interactive=False)

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

    with gr.Row() as button_row2:
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
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
            maximum=4096,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox] + btn_list)

    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, None, states + chatbots + [textbox] + btn_list
        )

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    send_file_btn.upload(
        send_file, states + model_selectors + [send_file_btn], states + chatbots + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    )

    return (
        states,
        model_selectors,
        chatbots,
        textbox,
        send_file_btn,
        button_row,
        button_row2,
        parameter_row,
    )
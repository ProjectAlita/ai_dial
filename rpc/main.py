from collections import deque
import itertools
from pydantic import ValidationError
from pylon.core.tools import web, log
import tiktoken  # pylint: disable=E0611,E0401

from tools import rpc_tools, constants
from ..models.integration_pd import IntegrationModel, AIDialSettings, AIModel
from ...integrations.models.pd.integration import SecretField


# def _get_redis_client():
#     return redis.Redis(
#         host=constants.REDIS_HOST, port=constants.REDIS_PORT,
#         db=constants.REDIS_AI_MODELS_DB, password=constants.REDIS_PASSWORD,
#         username=constants.REDIS_USER
#         )

def num_tokens_from_messages(messages: list, model: str) -> int:
    """Return the number of tokens used by a list of messages.
    See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        log.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        log.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        log.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        tokens_per_message = 4
        tokens_per_name = -1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    # num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def limit_conversation(
        conversation: dict, model_name: str, max_response_tokens: int, token_limit: int
        ) -> list:
        limited_conversation = []
        remaining_tokens = token_limit - max_response_tokens
        remaining_tokens -= 3  # every reply is primed with <|start|>assistant<|message|>

        context_tokens = num_tokens_from_messages(conversation['context'], model_name)
        remaining_tokens -= context_tokens
        limited_conversation.extend(conversation['context'])

        if remaining_tokens < 0:
            return limited_conversation

        input_tokens = num_tokens_from_messages(conversation['input'], model_name)
        remaining_tokens -= input_tokens
        if remaining_tokens < 0:
            return limited_conversation

        final_examples = []
        for example in conversation['examples']:
            example_tokens = num_tokens_from_messages([example], model_name)
            remaining_tokens -= example_tokens
            if remaining_tokens < 0:
                if len(final_examples) % 2:
                    final_examples.pop()  # remove incomplete example if present
                return limited_conversation + final_examples + conversation['input']
            final_examples.append(example)

        limited_conversation.extend(final_examples)

        final_history = deque()
        for message in reversed(conversation['chat_history']):
            message_tokens = num_tokens_from_messages([message], model_name)
            remaining_tokens -= message_tokens
            if remaining_tokens < 0:
                return limited_conversation + list(final_history) + conversation['input']
            final_history.appendleft(message)
        limited_conversation.extend(final_history)

        limited_conversation.extend(conversation['input'])
        return limited_conversation


def prepare_conversation(
        prompt_struct: dict, model_name: str, max_response_tokens: int, token_limit: int,
        check_limits: bool = True
        ) -> list:
    conversation = {
        'context': [],
        'examples': [],
        'chat_history': [],
        'input': []
    }

    if prompt_struct.get('context'):
        conversation['context'].append({
            "role": "system",
            "content": prompt_struct['context']
        })
    if prompt_struct.get('examples'):
        for example in prompt_struct['examples']:
            conversation['examples'].append({
                "role": "system",
                "name": "example_user",
                "content": example['input']
            })
            conversation['examples'].append({
                "role": "system",
                "name": "example_assistant",
                "content": example['output']
            })
    if prompt_struct.get('chat_history'):
        for message in prompt_struct['chat_history']:
            conversation['chat_history'].append({
                "role": "user" if message['role'] == 'user' else "assistant",
                "content": message['content']
            })
    if prompt_struct.get('prompt'):
        conversation['input'].append({
            "role": "user",
            "content": prompt_struct['prompt']
        })

    # conversation = context + examples + chat_history + input_

    # conv_history_tokens = num_tokens_from_messages(conversation, model_name)

    # while conv_history_tokens + max_response_tokens >= token_limit:
    #     if chat_history:
    #         del chat_history[0]
    #     elif examples:
    #         del examples[0:2]
    #     conversation = context + examples + chat_history + input_
    #     conv_history_tokens = num_tokens_from_messages(conversation, model_name)

    if check_limits:
        return limit_conversation(conversation, model_name, max_response_tokens, token_limit)

    return conversation['context'] + conversation['examples'] + conversation['chat_history'] + conversation['input']


def prepare_result(response):
    structured_result = {'messages': []}
    attachments = []
    if response['choices'][0]['message'].get('content'):
        structured_result['messages'].append({
            'type': 'text',
            'content': response['choices'][0]['message']['content']
        })
    else:
        attachments += response['choices'][0]['message'].get('custom_content', {}).get('attachments', [])
    attachments += response['choices'][0].get('custom_content', {}).get('attachments', [])
    for attachment in attachments:
        if 'image' in attachment.get('type', ''):
            structured_result['messages'].append({
                'type': 'image',
                'content': attachment
            })
        if 'text' in attachment.get('type', '') or not attachment.get('type'):
            content = attachment['title'] + '\n\n' if attachment.get('title') else ''
            content += attachment['data'] if attachment.get('data') else ''
            content += '\n\n' + 'Reference URL: ' + attachment['reference_url'] if attachment.get('reference_url') else ''
            structured_result['messages'].append({
                'type': 'text',
                'content': content
            })
    return structured_result


class RPC:
    integration_name = 'ai_dial'

    @web.rpc(f'{integration_name}__predict')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def predict(self, project_id, settings, prompt_struct):
        """ Predict function """
        import openai

        try:
            settings = IntegrationModel.parse_obj(settings)
        except ValidationError as e:
            return {"ok": False, "error": e}

        try:
            api_key = SecretField.parse_obj(settings.api_token).unsecret(project_id)
            openai.api_key = api_key
            openai.api_type = settings.api_type
            openai.api_base = settings.api_base
            openai.api_version = settings.api_version

            token_limit = settings.token_limit
            conversation = prepare_conversation(
                prompt_struct, settings.model_name, settings.max_tokens, token_limit)

            response = openai.ChatCompletion.create(
                deployment_id=settings.model_name,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                top_p=settings.top_p,
                messages=conversation
            )

            result = prepare_result(response)

        except Exception as e:
            log.error(str(e))
            return {"ok": False, "error": f"{str(e)}"}

        return {"ok": True, "response": result}

    @web.rpc(f'{integration_name}__parse_settings')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def parse_settings(self, settings):
        try:
            settings = AIDialSettings.parse_obj(settings)
        except ValidationError as e:
            return {"ok": False, "error": e}
        return {"ok": True, "item": settings}

    # @web.rpc(f'{integration_name}_get_models', 'get_models')
    # @rpc_tools.wrap_exceptions(RuntimeError)
    # def get_models(self):
    #     _rc = _get_redis_client()
    #     models = _rc.get(name=RPC.integration_name)
    #     return json.loads(models) if models else []

    @web.rpc(f'{integration_name}_set_models', 'set_models')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def set_models(self, payload: dict):
        import openai

        api_key = SecretField.parse_obj(payload['settings'].get('api_token', {})).unsecret(payload.get('project_id'))
        openai.api_key = api_key
        openai.api_type = payload['settings'].get('api_type')
        openai.api_base = payload['settings'].get('api_base')
        openai.api_version = payload['settings'].get('api_version')
        try:
            models = openai.Model.list()
        except Exception as e:
            log.error(str(e))
            models = []
        if models:
            models = models.get('data', [])
            models = [AIModel(**model).dict() for model in models]
        #     _rc = _get_redis_client()
        #     _rc.set(name=payload['name'], value=json.dumps(models))
        #     log.info(f'List of models for {payload["name"]} saved')
        return models

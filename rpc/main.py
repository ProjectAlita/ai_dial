import itertools
from pydantic import ValidationError
from pylon.core.tools import web, log  # pylint: disable=E0611,E0401
import tiktoken

from tools import rpc_tools, constants
from ..models.integration_pd import IntegrationModel, AIDialSettings, AIModel
from ...integrations.models.pd.integration import SecretField


# def _get_redis_client():
#     return redis.Redis(
#         host=constants.REDIS_HOST, port=constants.REDIS_PORT,
#         db=constants.REDIS_AI_MODELS_DB, password=constants.REDIS_PASSWORD,
#         username=constants.REDIS_USER
#         )


def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        log.warning(f"Warning: model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_name = -1  # if there's a name, the role is omitted
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def _prepare_conversation(prompt_struct, model_name, max_response_tokens, token_limit):
    conversation = []
    context = []
    examples = []
    chat_history = []
    input_ = []

    if prompt_struct.get('context'):
        context.append({
            "role": "system",
            "content": prompt_struct['context']
        })
    if prompt_struct.get('examples'):
        for example in prompt_struct['examples']:
            examples.append({
                "role": "system",
                "name": "example_user",
                "content": example['input']
            })
            examples.append({
                "role": "system",
                "name": "example_assistant",
                "content": example['output']
            })
    if prompt_struct.get('chat_history'):
        for message in prompt_struct['chat_history']:
            if message['role'] == 'user':
                chat_history.append({
                    "role": "user",
                    "content": message['content']
                })
            if message['role'] == 'ai':
                chat_history.append({
                    "role": "assistant",
                    "content": message['content']
                })
    if prompt_struct.get('prompt'):
        input_.append({
            "role": "user",
            "content": prompt_struct['prompt']
        })

    conversation = context + examples + chat_history + input_

    conv_history_tokens = num_tokens_from_messages(conversation, model_name)

    while conv_history_tokens + max_response_tokens >= token_limit:
        if chat_history:
            del chat_history[0]
        elif examples:
            del examples[0:2]
        conversation = context + examples + chat_history + input_
        conv_history_tokens = num_tokens_from_messages(conversation, model_name)

    return conversation

def _prepare_result(response):
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
            conversation = _prepare_conversation(
                prompt_struct, settings.model_name, settings.max_tokens, token_limit)

            response = openai.ChatCompletion.create(
                deployment_id=settings.model_name,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                top_p=settings.top_p,
                messages=conversation
            )

            result = _prepare_result(response)

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

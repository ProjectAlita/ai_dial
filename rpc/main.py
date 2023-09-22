from pydantic import ValidationError
from pylon.core.tools import web, log  # pylint: disable=E0611,E0401

from tools import rpc_tools, constants
from ..models.integration_pd import IntegrationModel, AIDialSettings
from ...integrations.models.pd.integration import SecretField


# def _get_redis_client():
#     return redis.Redis(
#         host=constants.REDIS_HOST, port=constants.REDIS_PORT,
#         db=constants.REDIS_AI_MODELS_DB, password=constants.REDIS_PASSWORD,
#         username=constants.REDIS_USER
#         )


def _prepare_conversation(prompt_struct):
    conversation = []
    if prompt_struct.get('context'):
        conversation.append({
            "role": "system",
            "content": prompt_struct['context']
        })
    if prompt_struct.get('examples'):
        for example in prompt_struct['examples']:
            conversation.append({
                "role": "user",
                "content": example['input']
            })
            conversation.append({
                "role": "assistant",
                "content": example['output']
            })
    if prompt_struct.get('prompt'):
        conversation.append({
            "role": "user",
            "content": prompt_struct['prompt']
        })

    return conversation

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

            conversation = _prepare_conversation(prompt_struct)

            response = openai.ChatCompletion.create(
                engine=settings.model_name,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                top_p=settings.top_p,
                messages=conversation
            )
            result = response['choices'][0]['message']['content']
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
        #     _rc = _get_redis_client()
        #     _rc.set(name=payload['name'], value=json.dumps(models))
        #     log.info(f'List of models for {payload["name"]} saved')
        return models

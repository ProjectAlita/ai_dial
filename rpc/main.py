from pydantic import ValidationError
from pylon.core.tools import web, log
from traceback import format_exc

from tools import rpc_tools, worker_client, this
from ..models.integration_pd import IntegrationModel, AIDialSettings, AIModel
from ..models.request_body import ChatCompletionRequestBody
from ..utils import predict_chat,  predict_chat_from_request
from ...integrations.models.pd.integration import SecretField


# def _get_redis_client():
#     return redis.Redis(
#         host=constants.REDIS_HOST, port=constants.REDIS_PORT,
#         db=constants.REDIS_AI_MODELS_DB, password=constants.REDIS_PASSWORD,
#         username=constants.REDIS_USER
#         )


class RPC:
    integration_name = 'ai_dial'

    @web.rpc(f'{integration_name}__predict')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def predict(self, project_id, settings, prompt_struct, format_response: bool = True, **kwargs):
        """ Predict function """
        try:
            result = predict_chat(
                project_id, settings, prompt_struct,
                format_response=format_response,
                **kwargs
            )
        except Exception as e:
            log.error(format_exc())
            return {"ok": False, "error": f"{type(e)}: {str(e)}"}

        return {"ok": True, "response": result}

    @web.rpc(f'{integration_name}__chat_completion')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def chat_completion(self, project_id, settings, request_data):
        """ Chat completion function """
        try:
            result = predict_chat_from_request(project_id, settings, request_data)
        except Exception as e:
            log.error(str(e))
            return {"ok": False, "error": f"{str(e)}"}

        return {"ok": True, "response": result}

    @web.rpc(f'{integration_name}__completion')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def completion(self, project_id, settings, request_data):
        """ Completion function """
        return {"ok": False, "error": "AI Dial supports only chat completion"}

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
        settings = {
            "api_base": payload["settings"]["api_base"],
            "api_version": payload["settings"]["api_version"],
        }
        #
        if isinstance(payload['settings'].get('api_token', {}), SecretField):
            token_field = payload['settings'].get('api_token')
        else:
            token_field = SecretField.parse_obj(
                payload['settings'].get('api_token', {})
            )
        #
        settings["api_token"] = token_field.unsecret(payload.get('project_id'))
        #
        raw_models = worker_client.ai_get_models(
            integration_name=this.module_name,
            settings=settings,
        )
        #
        return [AIModel(**model).dict() for model in raw_models]

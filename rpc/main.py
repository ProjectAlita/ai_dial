from pydantic import ValidationError
from pylon.core.tools import web, log
from traceback import format_exc

from tools import rpc_tools
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
        api_key = SecretField.parse_obj(payload['settings'].get('api_token', {})).unsecret(payload.get('project_id'))
        api_type = payload['settings'].get('api_type')
        api_base = payload['settings'].get('api_base')
        api_version = payload['settings'].get('api_version')
        try:
            # openai < 1.0.0
            from openai import Model
            models = Model.list(
                api_key=api_key, api_base=api_base, api_type=api_type, api_version=api_version
                )
        except Exception as e:
            # openai >= 1.0.0
            try:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    base_url=api_base,
                    api_version=api_version,
                    api_key=api_key,
                    # api_type is removed in openai >= 1.0.0
                )
                models = client.models.list()
            except Exception as e:
                log.error(str(e))
                models = []
        if models:
            try:
                models = models.get('data', [])
                models = [AIModel(**model).dict() for model in models]
            except:
                try:
                    models = [AIModel(id=model.id, name=model.name, capabilities=model.capabilities, token_limit=model.token_limit).dict() for model in models]
                except:
                    client_models = models
                    models = []
                    #
                    for model in client_models:
                        model_id = model.id
                        model_name = model.model
                        #
                        model_capabilities = {
                            "completion": model.capabilities["completion"],
                            "chat_completion": model.capabilities["chat_completion"],
                            "embeddings": model.capabilities["embeddings"],
                        }
                        #
                        try:
                            model_token_limit = model.limits["max_total_tokens"]
                        except:  # pylint: disable=W0702
                            model_token_limit = None
                        #
                        if model_token_limit is None:
                            ai_model = AIModel(
                                id=model_id,
                                name=model_name,
                                capabilities=model_capabilities,
                            ).dict()
                        else:
                            ai_model = AIModel(
                                id=model_id,
                                name=model_name,
                                capabilities=model_capabilities,
                                token_limit=model_token_limit,
                            ).dict()
                        #
                        models.append(ai_model)
        #
        return models

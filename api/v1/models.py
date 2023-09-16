from flask import request
from pydantic import ValidationError
from tools import api_tools

from ...models.integration_pd import IntegrationModel


class ProjectAPI(api_tools.APIModeHandler):
    # def get(self, project_id, **kwargs):
    #     models: list = self.module.get_models()
    #     return models

    def post(self, project_id, **kwargs):
        try:
            settings = IntegrationModel.parse_obj(request.json)
        except ValidationError as e:
            return e.errors(), 400

        check_connection_response = settings.check_connection()
        if check_connection_response is not True:
            return [{'loc': ['check_connection'], 'msg': check_connection_response}], 400

        models = settings.refresh_models(project_id)
        return models, 200


class AdminAPI(api_tools.APIModeHandler):
    ...


class API(api_tools.APIBase):
    url_params = [
        '<string:mode>/<int:project_id>',
        '<int:project_id>'
    ]

    mode_handlers = {
        'default': ProjectAPI,
        'administration': AdminAPI,
    }

#!/usr/bin/python3
# coding=utf-8

#   Copyright 2024 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Method """

import json

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401,W0611

from tool import worker_client  # pylint: disable=E0401


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    #
    # AI
    #

    @web.method()
    def ai_check_settings(  # pylint: disable=R0913
            self, settings,
        ):
        """ Check integration settings/test connection """
        #
        settings = json.loads(json.dumps(settings))
        #
        target_kwargs = {
            "azure_endpoint": settings["api_base"],
            "api_version": settings["api_version"],
            "api_key": settings["api_token"],
        }
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.chat_models.azure.AzureChatOpenAI",
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": "client._client",
            },
            "target_io_bound": True,
            #
            "method": "check_settings",
            "method_args": None,
            "method_kwargs": None,
        }
        #
        return result

    @web.method()
    def ai_get_models(  # pylint: disable=R0913
            self, settings,
        ):
        """ Get model list """
        #
        settings = json.loads(json.dumps(settings))
        #
        target_kwargs = {
            "azure_endpoint": settings["api_base"],
            "api_version": settings["api_version"],
            "api_key": settings["api_token"],
        }
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.chat_models.azure.AzureChatOpenAI",
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": "client._client",
            },
            "target_io_bound": True,
            #
            "method": "get_models",
            "method_args": None,
            "method_kwargs": None,
        }
        #
        return result

    @web.method()
    def count_tokens(  # pylint: disable=R0913
            self, settings, data,
        ):
        """ Count input/output/data tokens """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        target_kwargs = {
            "model": settings.merged_settings["model_name"],
            #
            **model_parameters,
            #
            "azure_endpoint": settings.merged_settings["api_base"],
            "api_version": settings.merged_settings["api_version"],
            "api_key": api_token,
        }
        #
        if isinstance(data, list):
            data = json.loads(json.dumps(data))
        #
        model_is_legacy_completion = False
        for model_data in settings.merged_settings["models"]:
            if model_data["name"] == settings.merged_settings["model_name"]:
                if not model_data["capabilities"]["chat_completion"]:
                    model_is_legacy_completion = True
        #
        target_class = "langchain_openai.chat_models.azure.AzureChatOpenAI"
        if model_is_legacy_completion:
            target_class = "langchain_openai.llms.azure.AzureOpenAI"
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": target_class,
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "count_tokens",
            "method_args": None,
            "method_kwargs": {
                "data": data,
            },
        }
        #
        return result

    #
    # LLM
    #

    @web.method()
    def llm_invoke(  # pylint: disable=R0913
            self, settings, text,
        ):
        """ Call model """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        target_kwargs = {
            "model": settings.merged_settings["model_name"],
            #
            **model_parameters,
            #
            "azure_endpoint": settings.merged_settings["api_base"],
            "api_version": settings.merged_settings["api_version"],
            "api_key": api_token,
        }
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.llms.azure.AzureOpenAI",
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "llm_invoke",
            "method_args": None,
            "method_kwargs": {
                "text": text,
            },
        }
        #
        return result

    @web.method()
    def llm_stream(  # pylint: disable=R0913
            self, settings, text, stream_id,
        ):
        """ Stream model """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        target_kwargs = {
            "model": settings.merged_settings["model_name"],
            #
            **model_parameters,
            #
            "azure_endpoint": settings.merged_settings["api_base"],
            "api_version": settings.merged_settings["api_version"],
            "api_key": api_token,
            #
            "streaming": True,
        }
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.llms.azure.AzureOpenAI",
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "llm_stream",
            "method_args": None,
            "method_kwargs": {
                "text": text,
                "stream_id": stream_id,
            },
        }
        #
        return result

    #
    # ChatModel
    #

    @web.method()
    def chat_model_invoke(  # pylint: disable=R0913
            self, settings, messages,
        ):
        """ Call model """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        target_kwargs = {
            "model": settings.merged_settings["model_name"],
            #
            **model_parameters,
            #
            "azure_endpoint": settings.merged_settings["api_base"],
            "api_version": settings.merged_settings["api_version"],
            "api_key": api_token,
        }
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.chat_models.azure.AzureChatOpenAI",
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "chat_invoke",
            "method_args": None,
            "method_kwargs": {
                "messages": json.loads(json.dumps(messages)),
            },
        }
        #
        return result

    @web.method()
    def chat_model_stream(  # pylint: disable=R0913
            self, settings, messages, stream_id,
        ):
        """ Stream model """
        #
        try:
            project_id = settings.integration.project_id
        except AttributeError:
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings.merged_settings["api_token"], project_id
        )
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings.merged_settings:
                model_parameters[param] = settings.merged_settings[param]
        #
        target_kwargs = {
            "model": settings.merged_settings["model_name"],
            #
            **model_parameters,
            #
            "azure_endpoint": settings.merged_settings["api_base"],
            "api_version": settings.merged_settings["api_version"],
            "api_key": api_token,
            #
            "streaming": True,
        }
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.chat_models.azure.AzureChatOpenAI",
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "chat_stream",
            "method_args": None,
            "method_kwargs": {
                "messages": json.loads(json.dumps(messages)),
                "stream_id": stream_id,
            },
        }
        #
        return result

    #
    # Embed
    #

    @web.method()
    def embed_documents(  # pylint: disable=R0913
            self, settings, texts,
        ):
        """ Make embeddings """
        #
        try:
            project_id = settings["integration_data"]["project_id"]
        except (AttributeError, KeyError):
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings["integration_data"]["settings"]["api_token"], project_id
        )
        #
        target_kwargs = {
            "model": settings["model_name"],
            #
            "azure_endpoint": settings["integration_data"]["settings"]["api_base"],
            "api_version": settings["integration_data"]["settings"]["api_version"],
            "api_key": api_token,
        }
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.embeddings.azure.AzureOpenAIEmbeddings",
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "embed_documents",
            "method_args": None,
            "method_kwargs": {
                "texts": texts,
            },
        }
        #
        return result

    @web.method()
    def embed_query(  # pylint: disable=R0913
            self, settings, text,
        ):
        """ Make embedding """
        #
        try:
            project_id = settings["integration_data"]["project_id"]
        except (AttributeError, KeyError):
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings["integration_data"]["settings"]["api_token"], project_id
        )
        #
        target_kwargs = {
            "model": settings["model_name"],
            #
            "azure_endpoint": settings["integration_data"]["settings"]["api_base"],
            "api_version": settings["integration_data"]["settings"]["api_version"],
            "api_key": api_token,
        }
        #
        result = {
            "routing_key": None,
            #
            "target": "plugins.ai_dial_worker.utils.ai.Helper",
            "target_args": None,
            "target_kwargs": {
                "target_class": "langchain_openai.embeddings.azure.AzureOpenAIEmbeddings",
                "target_args": None,
                "target_kwargs": target_kwargs,
                "client_attr": None,
            },
            "target_io_bound": True,
            #
            "method": "embed_query",
            "method_args": None,
            "method_kwargs": {
                "text": text,
            },
        }
        #
        return result

    #
    # Indexer
    #

    @web.method()
    def indexer_config(  # pylint: disable=R0913
            self, settings, model,
        ):
        """ Make indexer config """
        #
        model_info = None
        #
        for item in settings["settings"]["models"]:
            if item["name"] == model:
                model_info = item
                break
        #
        if model_info is None:
            raise RuntimeError(f"No model info found: {model}")
        #
        try:
            project_id = settings["project_id"]
        except (AttributeError, KeyError):
            project_id = None
        #
        api_token = worker_client.unsecret_data(
            settings["settings"]["api_token"], project_id
        )
        #
        auth_kwargs = {
            "azure_endpoint": settings["settings"]["api_base"],
            "api_version": settings["settings"]["api_version"],
            "api_key": api_token,
        }
        #
        if model_info["capabilities"]["embeddings"]:
            return {
                "embedding_model": "langchain_openai.embeddings.azure.AzureOpenAIEmbeddings",
                "embedding_model_params": {
                    "model": model,
                    #
                    **auth_kwargs,
                },
            }
        #
        model_parameters = {}
        #
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in settings["settings"]:
                model_parameters[param] = settings["settings"][param]
        #
        if not model_info["capabilities"]["chat_completion"]:
            return {
                "ai_model": "langchain_openai.llms.azure.AzureOpenAI",
                "ai_model_params": {
                    "model": model,
                    #
                    **model_parameters,
                    **auth_kwargs,
                },
            }
        #
        return {
            "ai_model": "langchain_openai.chat_models.azure.AzureChatOpenAI",
            "ai_model_params": {
                "model": model,
                #
                **model_parameters,
                **auth_kwargs,
            },
        }

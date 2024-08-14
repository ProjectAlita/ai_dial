#!/usr/bin/python3
# coding=utf-8

#   Copyright 2021 getcarrier.io
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

""" Module """
import json
from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import module  # pylint: disable=E0611,E0401

from tools import VaultClient, worker_client  # pylint: disable=E0611,E0401

from .models.integration_pd import IntegrationModel


TOKEN_LIMITS = {
    'text-embedding-ada-002': None,
    'gpt-35-turbo': 4096,
    'gpt-35-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-4-turbo': 128000,
    'gpt-4o': 128000,
    'gpt-4o-mini': 128000,
    'chat-bison@001': 8192,
    'ai21.j2-grande-instruct': 8191,
    'ai21.j2-jumbo-instruct': 8191,
    'anthropic.claude-instant-v1': 100000,
    'anthropic.claude-v1': 100000,
    'anthropic.claude-v2': 100000,
    'stability.stable-diffusion-xl': 77
}


class Module(module.ModuleModel):
    """ Task module """

    def __init__(self, context, descriptor):
        self.context = context
        self.descriptor = descriptor

    def init(self):
        """ Init module """
        log.info('Initializing AI Dial module')
        SECTION_NAME = 'ai'
        #
        self.descriptor.init_all()
        #
        self.context.rpc_manager.call.integrations_register_section(
            name=SECTION_NAME,
            integration_description='Manage ai integrations',
        )
        self.context.rpc_manager.call.integrations_register(
            name=self.descriptor.name,
            section=SECTION_NAME,
            settings_model=IntegrationModel,
        )
        #
        vault_client = VaultClient()
        secrets = vault_client.get_all_secrets()
        if 'ai_dial_token_limits' not in secrets:
            secrets['ai_dial_token_limits'] = json.dumps(TOKEN_LIMITS)
            vault_client.set_secrets(secrets)
        #
        worker_client.register_integration(
            integration_name=self.descriptor.name,
            #
            ai_check_settings_callback=self.ai_check_settings,
            ai_get_models_callback=self.ai_get_models,
            ai_count_tokens_callback=self.count_tokens,
            #
            llm_invoke_callback=self.llm_invoke,
            llm_stream_callback=self.llm_stream,
            #
            chat_model_invoke_callback=self.chat_model_invoke,
            chat_model_stream_callback=self.chat_model_stream,
            #
            embed_documents_callback=self.embed_documents,
            embed_query_callback=self.embed_query,
            #
            indexer_config_callback=self.indexer_config,
        )

    def deinit(self):  # pylint: disable=R0201
        """ De-init module """
        log.info('De-initializing')
        #
        self.descriptor.deinit_all()

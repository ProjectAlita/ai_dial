#!/usr/bin/python3
# coding=utf-8

#   Copyright 2025 EPAM Systems
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

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401,W0611


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    #
    # AI Model Params
    #

    @web.method()
    def override_ai_model_params(  # pylint: disable=R0913
            self, model, parameters,
        ):
        """ Apply required changes """
        result = parameters.copy()
        #
        if model in self.descriptor.config.get("apply_o1_overrides_for", []):
            remap_keys = {
                "max_tokens": "max_completion_tokens",
            }
            #
            drop_keys = [
                "temperature",
                "top_p",
            ]
            #
            for key, target in remap_keys.items():
                if key in result:
                    result[target] = result.pop(key)
            #
            for key in drop_keys:
                if key in result:
                    result.pop(key)
        #
        return result

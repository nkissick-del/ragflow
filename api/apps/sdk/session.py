#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from .session_impl.chat import register_chat_routes
from .session_impl.agent import register_agent_routes
from .session_impl.bot import register_bot_routes
from .session_impl.session_ops import register_ops_routes


# manager is injected by api/apps/__init__.py
def setup_routes(manager):
    register_chat_routes(manager)
    register_agent_routes(manager)
    register_bot_routes(manager)
    register_ops_routes(manager)

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ToonDB Error Types
"""


class ToonDBError(Exception):
    """Base exception for ToonDB errors."""
    pass


class ConnectionError(ToonDBError):
    """Failed to connect to ToonDB server or open database."""
    pass


class TransactionError(ToonDBError):
    """Transaction-related error."""
    pass


class ProtocolError(ToonDBError):
    """Wire protocol error."""
    pass


class DatabaseError(ToonDBError):
    """Database operation error."""
    pass

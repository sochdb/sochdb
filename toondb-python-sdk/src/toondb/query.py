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
Query Builder for ToonDB.
"""

from typing import List, Dict, Any, Optional, Union
from .ipc_client import IpcClient

class Query:
    """
    Fluent query builder for ToonDB.
    
    Example:
        db.query("users/") \
          .limit(10) \
          .select(["name", "email"]) \
          .execute()
    """
    
    def __init__(self, client: IpcClient, path_prefix: str):
        self._client = client
        self._path_prefix = path_prefix
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._columns: Optional[List[str]] = None
        
    def limit(self, n: int) -> "Query":
        """Limit the number of results."""
        self._limit = n
        return self
        
    def offset(self, n: int) -> "Query":
        """Skip the first n results."""
        self._offset = n
        return self
        
    def select(self, columns: List[str]) -> "Query":
        """Select specific columns to return."""
        self._columns = columns
        return self
        
    def execute(self) -> str:
        """
        Execute the query and return results as TOON string.
        
        Returns:
            TOON formatted string (e.g., "result[N]{cols}: row1; row2")
        """
        return self._client.query(
            self._path_prefix,
            limit=self._limit,
            offset=self._offset,
            columns=self._columns
        )
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        Execute and parse results into a list of dictionaries.
        
        Note: This is a simple parser for the TOON format returned by the server.
        For production, a robust TOON parser is recommended.
        """
        toon_str = self.execute()
        return self._parse_toon(toon_str)

    def _parse_toon(self, toon_str: str) -> List[Dict[str, Any]]:
        """Simple TOON parser."""
        if not toon_str or toon_str == "[]":
            return []
            
        # Format: result[N]{col1,col2}: val1,val2; val3,val4
        try:
            header_end = toon_str.find(":")
            if header_end == -1:
                return []
                
            header = toon_str[:header_end]
            body = toon_str[header_end+1:].strip()
            
            # Parse header: result[N]{cols}
            cols_start = header.find("{")
            cols_end = header.find("}")
            if cols_start == -1 or cols_end == -1:
                return []
                
            cols_str = header[cols_start+1:cols_end]
            columns = [c.strip() for c in cols_str.split(",")]
            
            if not body:
                return []
                
            rows = []
            # Split by semicolon for rows
            # Note: This is naive and will break if values contain semicolons not in quotes
            # A real parser would handle state
            row_strs = body.split(";")
            
            for row_str in row_strs:
                if not row_str.strip():
                    continue
                    
                # Split by comma for values
                # Again, naive splitting
                vals = row_str.strip().split(",")
                
                row = {}
                for i, col in enumerate(columns):
                    if i < len(vals):
                        val = vals[i].strip()
                        # Basic type inference
                        if val.isdigit():
                            row[col] = int(val)
                        elif val.replace(".", "", 1).isdigit():
                            row[col] = float(val)
                        elif val == "T":
                            row[col] = True
                        elif val == "F":
                            row[col] = False
                        elif val == "∅":
                            row[col] = None
                        elif val.startswith('"') and val.endswith('"'):
                            row[col] = val[1:-1]
                        else:
                            row[col] = val
                rows.append(row)
                
            return rows
        except Exception:
            return []

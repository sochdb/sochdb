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
SQL Engine for ToonDB Python SDK.

Provides SQL support on top of the KV storage backend.
Tables are stored as:
  - Schema: _sql/tables/{table_name}/schema -> JSON schema definition
  - Rows: _sql/tables/{table_name}/rows/{row_id} -> JSON row data
  - Indexes: _sql/tables/{table_name}/indexes/{index_name} -> index data
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from .query import SQLQueryResult


@dataclass
class Column:
    """SQL column definition."""
    name: str
    type: str  # INT, TEXT, FLOAT, BOOL, BLOB
    nullable: bool = True
    primary_key: bool = False
    default: Any = None


@dataclass
class TableSchema:
    """SQL table schema."""
    name: str
    columns: List[Column] = field(default_factory=list)
    primary_key: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "columns": [
                {
                    "name": c.name,
                    "type": c.type,
                    "nullable": c.nullable,
                    "primary_key": c.primary_key,
                    "default": c.default
                }
                for c in self.columns
            ],
            "primary_key": self.primary_key
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TableSchema":
        columns = [
            Column(
                name=c["name"],
                type=c["type"],
                nullable=c.get("nullable", True),
                primary_key=c.get("primary_key", False),
                default=c.get("default")
            )
            for c in data.get("columns", [])
        ]
        return cls(
            name=data["name"],
            columns=columns,
            primary_key=data.get("primary_key")
        )


class SQLParser:
    """Simple SQL parser for basic DDL and DML."""
    
    @staticmethod
    def parse(sql: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse SQL and return (operation, parsed_data).
        
        Returns:
            Tuple of (operation_type, parsed_info)
            operation_type: CREATE_TABLE, DROP_TABLE, INSERT, SELECT, UPDATE, DELETE
        """
        sql = sql.strip()
        upper = sql.upper()
        
        if upper.startswith("CREATE TABLE"):
            return SQLParser._parse_create_table(sql)
        elif upper.startswith("DROP TABLE"):
            return SQLParser._parse_drop_table(sql)
        elif upper.startswith("INSERT"):
            return SQLParser._parse_insert(sql)
        elif upper.startswith("SELECT"):
            return SQLParser._parse_select(sql)
        elif upper.startswith("UPDATE"):
            return SQLParser._parse_update(sql)
        elif upper.startswith("DELETE"):
            return SQLParser._parse_delete(sql)
        else:
            raise ValueError(f"Unsupported SQL statement: {sql[:50]}")
    
    @staticmethod
    def _parse_create_table(sql: str) -> Tuple[str, Dict]:
        """Parse CREATE TABLE statement."""
        # CREATE TABLE table_name (col1 TYPE, col2 TYPE, ...)
        match = re.match(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*)\)',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        if not match:
            raise ValueError(f"Invalid CREATE TABLE: {sql}")
        
        table_name = match.group(1)
        cols_str = match.group(2)
        
        columns = []
        primary_key = None
        
        # Split by comma, but not inside parentheses
        col_defs = SQLParser._split_columns(cols_str)
        
        for col_def in col_defs:
            col_def = col_def.strip()
            if not col_def:
                continue
            
            # Check for PRIMARY KEY constraint
            if col_def.upper().startswith("PRIMARY KEY"):
                pk_match = re.match(r'PRIMARY\s+KEY\s*\((\w+)\)', col_def, re.IGNORECASE)
                if pk_match:
                    primary_key = pk_match.group(1)
                continue
            
            # Parse column: name TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT value]
            parts = col_def.split()
            if len(parts) < 2:
                continue
            
            col_name = parts[0]
            col_type = parts[1].upper()
            
            # Normalize types
            if col_type in ("INTEGER", "INT", "BIGINT", "SMALLINT"):
                col_type = "INT"
            elif col_type in ("VARCHAR", "CHAR", "STRING", "TEXT"):
                col_type = "TEXT"
            elif col_type in ("REAL", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC"):
                col_type = "FLOAT"
            elif col_type in ("BOOLEAN", "BOOL"):
                col_type = "BOOL"
            elif col_type in ("BLOB", "BYTES", "BINARY"):
                col_type = "BLOB"
            
            col_upper = col_def.upper()
            is_pk = "PRIMARY KEY" in col_upper
            nullable = "NOT NULL" not in col_upper
            
            if is_pk:
                primary_key = col_name
            
            columns.append(Column(
                name=col_name,
                type=col_type,
                nullable=nullable,
                primary_key=is_pk
            ))
        
        return "CREATE_TABLE", {
            "table": table_name,
            "columns": columns,
            "primary_key": primary_key
        }
    
    @staticmethod
    def _split_columns(cols_str: str) -> List[str]:
        """Split column definitions, handling parentheses."""
        result = []
        current = ""
        depth = 0
        
        for char in cols_str:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                result.append(current)
                current = ""
            else:
                current += char
        
        if current.strip():
            result.append(current)
        
        return result
    
    @staticmethod
    def _parse_drop_table(sql: str) -> Tuple[str, Dict]:
        """Parse DROP TABLE statement."""
        match = re.match(
            r'DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)',
            sql,
            re.IGNORECASE
        )
        if not match:
            raise ValueError(f"Invalid DROP TABLE: {sql}")
        
        return "DROP_TABLE", {"table": match.group(1)}
    
    @staticmethod
    def _parse_insert(sql: str) -> Tuple[str, Dict]:
        """Parse INSERT statement."""
        # INSERT INTO table (col1, col2) VALUES (val1, val2)
        # or INSERT INTO table VALUES (val1, val2)
        
        # With column names
        match = re.match(
            r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\((.+)\)',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if match:
            table = match.group(1)
            columns = [c.strip() for c in match.group(2).split(',')]
            values = SQLParser._parse_values(match.group(3))
            return "INSERT", {"table": table, "columns": columns, "values": values}
        
        # Without column names
        match = re.match(
            r'INSERT\s+INTO\s+(\w+)\s+VALUES\s*\((.+)\)',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if match:
            table = match.group(1)
            values = SQLParser._parse_values(match.group(2))
            return "INSERT", {"table": table, "columns": None, "values": values}
        
        raise ValueError(f"Invalid INSERT: {sql}")
    
    @staticmethod
    def _parse_values(values_str: str) -> List[Any]:
        """Parse value list from VALUES clause."""
        values = []
        current = ""
        in_string = False
        string_char = None
        
        for char in values_str:
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
                current += char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
                current += char
            elif char == ',' and not in_string:
                values.append(SQLParser._parse_value(current.strip()))
                current = ""
            else:
                current += char
        
        if current.strip():
            values.append(SQLParser._parse_value(current.strip()))
        
        return values
    
    @staticmethod
    def _parse_value(val_str: str) -> Any:
        """Parse a single value."""
        val_str = val_str.strip()
        
        if not val_str or val_str.upper() == "NULL":
            return None
        
        # String literals
        if (val_str.startswith("'") and val_str.endswith("'")) or \
           (val_str.startswith('"') and val_str.endswith('"')):
            return val_str[1:-1]
        
        # Boolean
        if val_str.upper() == "TRUE":
            return True
        if val_str.upper() == "FALSE":
            return False
        
        # Numbers
        try:
            if '.' in val_str:
                return float(val_str)
            return int(val_str)
        except ValueError:
            return val_str
    
    @staticmethod
    def _parse_select(sql: str) -> Tuple[str, Dict]:
        """Parse SELECT statement."""
        # SELECT cols FROM table [WHERE ...] [ORDER BY ...] [LIMIT ...]
        
        # Extract main parts using regex
        pattern = r'''
            SELECT\s+(.+?)           # columns
            \s+FROM\s+(\w+)          # table
            (?:\s+WHERE\s+(.+?))?    # optional WHERE
            (?:\s+ORDER\s+BY\s+(.+?))?  # optional ORDER BY
            (?:\s+LIMIT\s+(\d+))?    # optional LIMIT
            (?:\s+OFFSET\s+(\d+))?   # optional OFFSET
            \s*$
        '''
        
        match = re.match(pattern, sql, re.IGNORECASE | re.DOTALL | re.VERBOSE)
        
        if not match:
            # Simpler pattern for basic SELECT
            simple_match = re.match(
                r'SELECT\s+(.+?)\s+FROM\s+(\w+)',
                sql,
                re.IGNORECASE | re.DOTALL
            )
            if not simple_match:
                raise ValueError(f"Invalid SELECT: {sql}")
            
            columns_str = simple_match.group(1)
            table = simple_match.group(2)
            
            # Parse rest of the query
            rest = sql[simple_match.end():].strip()
            where_clause = None
            order_by = None
            limit = None
            offset = None
            
            # Extract WHERE
            where_match = re.search(r'\bWHERE\s+(.+?)(?:\s+ORDER|\s+LIMIT|\s+OFFSET|$)', rest, re.IGNORECASE)
            if where_match:
                where_clause = where_match.group(1).strip()
            
            # Extract ORDER BY
            order_match = re.search(r'\bORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s+OFFSET|$)', rest, re.IGNORECASE)
            if order_match:
                order_by = order_match.group(1).strip()
            
            # Extract LIMIT
            limit_match = re.search(r'\bLIMIT\s+(\d+)', rest, re.IGNORECASE)
            if limit_match:
                limit = int(limit_match.group(1))
            
            # Extract OFFSET
            offset_match = re.search(r'\bOFFSET\s+(\d+)', rest, re.IGNORECASE)
            if offset_match:
                offset = int(offset_match.group(1))
        else:
            columns_str = match.group(1)
            table = match.group(2)
            where_clause = match.group(3)
            order_by = match.group(4)
            limit = int(match.group(5)) if match.group(5) else None
            offset = int(match.group(6)) if match.group(6) else None
        
        # Parse columns
        if columns_str.strip() == "*":
            columns = ["*"]
        else:
            columns = [c.strip() for c in columns_str.split(',')]
        
        # Parse WHERE clause
        conditions = []
        if where_clause:
            conditions = SQLParser._parse_where(where_clause)
        
        # Parse ORDER BY
        order = []
        if order_by:
            for part in order_by.split(','):
                part = part.strip()
                if part.upper().endswith(" DESC"):
                    order.append((part[:-5].strip(), "DESC"))
                elif part.upper().endswith(" ASC"):
                    order.append((part[:-4].strip(), "ASC"))
                else:
                    order.append((part, "ASC"))
        
        return "SELECT", {
            "table": table,
            "columns": columns,
            "where": conditions,
            "order_by": order,
            "limit": limit,
            "offset": offset
        }
    
    @staticmethod
    def _parse_where(where_clause: str) -> List[Tuple[str, str, Any]]:
        """Parse WHERE clause into list of (column, operator, value)."""
        conditions = []
        
        # Split by AND (simple case, doesn't handle nested OR)
        parts = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            
            # Match: column operator value
            match = re.match(r'(\w+)\s*(=|!=|<>|>=|<=|>|<|LIKE|NOT\s+LIKE)\s*(.+)', part, re.IGNORECASE)
            if match:
                col = match.group(1)
                op = match.group(2).upper().replace(" ", "_")
                if op == "<>":
                    op = "!="
                val = SQLParser._parse_value(match.group(3))
                conditions.append((col, op, val))
        
        return conditions
    
    @staticmethod
    def _parse_update(sql: str) -> Tuple[str, Dict]:
        """Parse UPDATE statement."""
        # UPDATE table SET col1=val1, col2=val2 [WHERE ...]
        match = re.match(
            r'UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if not match:
            raise ValueError(f"Invalid UPDATE: {sql}")
        
        table = match.group(1)
        set_clause = match.group(2)
        where_clause = match.group(3)
        
        # Parse SET clause
        updates = {}
        for part in set_clause.split(','):
            eq_match = re.match(r'\s*(\w+)\s*=\s*(.+)\s*', part)
            if eq_match:
                col = eq_match.group(1)
                val = SQLParser._parse_value(eq_match.group(2))
                updates[col] = val
        
        conditions = []
        if where_clause:
            conditions = SQLParser._parse_where(where_clause)
        
        return "UPDATE", {"table": table, "updates": updates, "where": conditions}
    
    @staticmethod
    def _parse_delete(sql: str) -> Tuple[str, Dict]:
        """Parse DELETE statement."""
        match = re.match(
            r'DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?$',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if not match:
            raise ValueError(f"Invalid DELETE: {sql}")
        
        table = match.group(1)
        where_clause = match.group(2)
        
        conditions = []
        if where_clause:
            conditions = SQLParser._parse_where(where_clause)
        
        return "DELETE", {"table": table, "where": conditions}


class SQLExecutor:
    """Execute SQL operations using the KV backend."""
    
    # Key prefixes for SQL data
    TABLE_PREFIX = b"_sql/tables/"
    SCHEMA_SUFFIX = b"/schema"
    ROWS_PREFIX = b"/rows/"
    
    def __init__(self, db):
        """Initialize with a Database instance."""
        self._db = db
    
    def execute(self, sql: str) -> SQLQueryResult:
        """Execute a SQL statement."""
        operation, data = SQLParser.parse(sql)
        
        if operation == "CREATE_TABLE":
            return self._create_table(data)
        elif operation == "DROP_TABLE":
            return self._drop_table(data)
        elif operation == "INSERT":
            return self._insert(data)
        elif operation == "SELECT":
            return self._select(data)
        elif operation == "UPDATE":
            return self._update(data)
        elif operation == "DELETE":
            return self._delete(data)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _schema_key(self, table: str) -> bytes:
        """Get the key for a table's schema."""
        return self.TABLE_PREFIX + table.encode() + self.SCHEMA_SUFFIX
    
    def _row_key(self, table: str, row_id: str) -> bytes:
        """Get the key for a specific row."""
        return self.TABLE_PREFIX + table.encode() + self.ROWS_PREFIX + row_id.encode()
    
    def _row_prefix(self, table: str) -> bytes:
        """Get the prefix for all rows in a table."""
        return self.TABLE_PREFIX + table.encode() + self.ROWS_PREFIX
    
    def _get_schema(self, table: str) -> Optional[TableSchema]:
        """Get table schema."""
        data = self._db.get(self._schema_key(table))
        if data is None:
            return None
        return TableSchema.from_dict(json.loads(data.decode()))
    
    def _create_table(self, data: Dict) -> SQLQueryResult:
        """Create a new table."""
        table = data["table"]
        columns = data["columns"]
        primary_key = data.get("primary_key")
        
        # Check if table exists
        if self._get_schema(table) is not None:
            raise ValueError(f"Table '{table}' already exists")
        
        schema = TableSchema(name=table, columns=columns, primary_key=primary_key)
        
        # Store schema
        self._db.put(
            self._schema_key(table),
            json.dumps(schema.to_dict()).encode()
        )
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=0)
    
    def _drop_table(self, data: Dict) -> SQLQueryResult:
        """Drop a table."""
        table = data["table"]
        
        # Delete all rows first
        prefix = self._row_prefix(table)
        rows_deleted = 0
        for key, _ in self._db.scan_prefix(prefix):
            self._db.delete(key)
            rows_deleted += 1
        
        # Delete schema
        self._db.delete(self._schema_key(table))
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=rows_deleted)
    
    def _insert(self, data: Dict) -> SQLQueryResult:
        """Insert a row."""
        table = data["table"]
        columns = data["columns"]
        values = data["values"]
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        # If no columns specified, use schema order
        if columns is None:
            columns = [c.name for c in schema.columns]
        
        if len(columns) != len(values):
            raise ValueError(f"Column count ({len(columns)}) doesn't match value count ({len(values)})")
        
        # Create row dict
        row = dict(zip(columns, values))
        
        # Generate row ID (use primary key value or UUID)
        if schema.primary_key and schema.primary_key in row:
            row_id = str(row[schema.primary_key])
        else:
            row_id = str(uuid.uuid4())
        
        # Add row ID to row data
        row["_id"] = row_id
        
        # Store row
        self._db.put(
            self._row_key(table, row_id),
            json.dumps(row).encode()
        )
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=1)
    
    def _select(self, data: Dict) -> SQLQueryResult:
        """Select rows."""
        table = data["table"]
        columns = data["columns"]
        conditions = data.get("where", [])
        order_by = data.get("order_by", [])
        limit = data.get("limit")
        offset = data.get("offset", 0)
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        # Get column names
        if columns == ["*"]:
            columns = [c.name for c in schema.columns]
        
        # Scan all rows
        prefix = self._row_prefix(table)
        rows = []
        
        for key, value in self._db.scan_prefix(prefix):
            row = json.loads(value.decode())
            
            # Apply WHERE conditions
            if self._matches_conditions(row, conditions):
                # Project columns
                projected = {col: row.get(col) for col in columns if col in row}
                rows.append(projected)
        
        # Apply ORDER BY
        if order_by:
            for col, direction in reversed(order_by):
                reverse = direction == "DESC"
                rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=reverse)
        
        # Apply OFFSET and LIMIT
        if offset:
            rows = rows[offset:]
        if limit:
            rows = rows[:limit]
        
        return SQLQueryResult(rows=rows, columns=columns, rows_affected=0)
    
    def _matches_conditions(self, row: Dict, conditions: List[Tuple]) -> bool:
        """Check if a row matches all conditions."""
        for col, op, val in conditions:
            row_val = row.get(col)
            
            if op == "=":
                if row_val != val:
                    return False
            elif op == "!=":
                if row_val == val:
                    return False
            elif op == ">":
                if row_val is None or row_val <= val:
                    return False
            elif op == ">=":
                if row_val is None or row_val < val:
                    return False
            elif op == "<":
                if row_val is None or row_val >= val:
                    return False
            elif op == "<=":
                if row_val is None or row_val > val:
                    return False
            elif op == "LIKE":
                if row_val is None:
                    return False
                # Convert SQL LIKE to regex
                pattern = val.replace("%", ".*").replace("_", ".")
                if not re.match(f"^{pattern}$", str(row_val), re.IGNORECASE):
                    return False
            elif op == "NOT_LIKE":
                if row_val is None:
                    return True
                pattern = val.replace("%", ".*").replace("_", ".")
                if re.match(f"^{pattern}$", str(row_val), re.IGNORECASE):
                    return False
        
        return True
    
    def _update(self, data: Dict) -> SQLQueryResult:
        """Update rows."""
        table = data["table"]
        updates = data["updates"]
        conditions = data.get("where", [])
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        # Scan all rows
        prefix = self._row_prefix(table)
        rows_affected = 0
        
        for key, value in self._db.scan_prefix(prefix):
            row = json.loads(value.decode())
            
            # Apply WHERE conditions
            if self._matches_conditions(row, conditions):
                # Apply updates
                for col, val in updates.items():
                    row[col] = val
                
                # Save updated row
                self._db.put(key, json.dumps(row).encode())
                rows_affected += 1
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=rows_affected)
    
    def _delete(self, data: Dict) -> SQLQueryResult:
        """Delete rows."""
        table = data["table"]
        conditions = data.get("where", [])
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        # Scan all rows
        prefix = self._row_prefix(table)
        rows_affected = 0
        keys_to_delete = []
        
        # Collect keys to delete (don't modify while iterating)
        for key, value in self._db.scan_prefix(prefix):
            row = json.loads(value.decode())
            
            # Apply WHERE conditions
            if self._matches_conditions(row, conditions):
                keys_to_delete.append(key)
        
        # Delete collected keys
        for key in keys_to_delete:
            self._db.delete(key)
            rows_affected += 1
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=rows_affected)

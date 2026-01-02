package toondb

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/google/uuid"
)

// SQLEngine handles SQL query execution using the IPC client
type SQLEngine struct {
	client *IPCClient
}

// NewSQLEngine creates a new SQL engine
func NewSQLEngine(client *IPCClient) *SQLEngine {
	return &SQLEngine{client: client}
}

// Column represents a table column definition
type Column struct {
	Name       string      `json:"name"`
	Type       string      `json:"type"`
	Nullable   bool        `json:"nullable"`
	PrimaryKey bool        `json:"primary_key"`
	Default    interface{} `json:"default,omitempty"`
}

// TableSchema represents a table schema
type TableSchema struct {
	Name       string   `json:"name"`
	Columns    []Column `json:"columns"`
	PrimaryKey string   `json:"primary_key,omitempty"`
}

// Key prefixes for SQL data
const (
	sqlTablePrefix  = "_sql/tables/"
	sqlSchemaSuffix = "/schema"
	sqlRowsPrefix   = "/rows/"
)

// Execute executes a SQL query
func (e *SQLEngine) Execute(sql string) (*SQLQueryResult, error) {
	sql = strings.TrimSpace(sql)
	upper := strings.ToUpper(sql)

	if strings.HasPrefix(upper, "CREATE TABLE") {
		return e.executeCreateTable(sql)
	} else if strings.HasPrefix(upper, "DROP TABLE") {
		return e.executeDropTable(sql)
	} else if strings.HasPrefix(upper, "INSERT") {
		return e.executeInsert(sql)
	} else if strings.HasPrefix(upper, "SELECT") {
		return e.executeSelect(sql)
	} else if strings.HasPrefix(upper, "UPDATE") {
		return e.executeUpdate(sql)
	} else if strings.HasPrefix(upper, "DELETE") {
		return e.executeDelete(sql)
	}

	return nil, fmt.Errorf("unsupported SQL statement: %s", sql[:min(50, len(sql))])
}

func (e *SQLEngine) schemaKey(table string) []byte {
	return []byte(sqlTablePrefix + table + sqlSchemaSuffix)
}

func (e *SQLEngine) rowKey(table, rowID string) []byte {
	return []byte(sqlTablePrefix + table + sqlRowsPrefix + rowID)
}

func (e *SQLEngine) rowPrefix(table string) []byte {
	return []byte(sqlTablePrefix + table + sqlRowsPrefix)
}

func (e *SQLEngine) getSchema(table string) (*TableSchema, error) {
	data, err := e.client.Get(e.schemaKey(table))
	if err != nil || data == nil {
		return nil, nil
	}

	var schema TableSchema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("failed to parse schema: %w", err)
	}

	return &schema, nil
}

func (e *SQLEngine) executeCreateTable(sql string) (*SQLQueryResult, error) {
	// Parse: CREATE TABLE [IF NOT EXISTS] table_name (col1 TYPE, col2 TYPE, ...)
	re := regexp.MustCompile(`(?i)CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*)\)`)
	matches := re.FindStringSubmatch(sql)
	if matches == nil {
		return nil, fmt.Errorf("invalid CREATE TABLE: %s", sql)
	}

	tableName := matches[1]
	colsStr := matches[2]

	// Check if table exists
	existing, err := e.getSchema(tableName)
	if err != nil {
		return nil, err
	}
	if existing != nil {
		return nil, fmt.Errorf("table '%s' already exists", tableName)
	}

	// Parse columns
	columns, primaryKey := parseColumns(colsStr)

	schema := TableSchema{
		Name:       tableName,
		Columns:    columns,
		PrimaryKey: primaryKey,
	}

	// Store schema
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize schema: %w", err)
	}

	if err := e.client.Put(e.schemaKey(tableName), schemaJSON); err != nil {
		return nil, fmt.Errorf("failed to store schema: %w", err)
	}

	return &SQLQueryResult{
		Rows:         []map[string]interface{}{},
		Columns:      []string{},
		RowsAffected: 0,
	}, nil
}

func (e *SQLEngine) executeDropTable(sql string) (*SQLQueryResult, error) {
	// Parse: DROP TABLE [IF EXISTS] table_name
	re := regexp.MustCompile(`(?i)DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)`)
	matches := re.FindStringSubmatch(sql)
	if matches == nil {
		return nil, fmt.Errorf("invalid DROP TABLE: %s", sql)
	}

	tableName := matches[1]

	// Delete all rows
	prefix := string(e.rowPrefix(tableName))
	results, err := e.client.Scan(prefix)
	if err != nil {
		return nil, fmt.Errorf("failed to scan rows: %w", err)
	}

	rowsDeleted := 0
	for _, result := range results {
		if err := e.client.Delete(result.Key); err != nil {
			return nil, fmt.Errorf("failed to delete row: %w", err)
		}
		rowsDeleted++
	}

	// Delete schema
	if err := e.client.Delete(e.schemaKey(tableName)); err != nil {
		return nil, fmt.Errorf("failed to delete schema: %w", err)
	}

	return &SQLQueryResult{
		Rows:         []map[string]interface{}{},
		Columns:      []string{},
		RowsAffected: rowsDeleted,
	}, nil
}

func (e *SQLEngine) executeInsert(sql string) (*SQLQueryResult, error) {
	// Parse: INSERT INTO table (cols) VALUES (vals) or INSERT INTO table VALUES (vals)
	re1 := regexp.MustCompile(`(?i)INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\((.+)\)`)
	matches := re1.FindStringSubmatch(sql)

	var tableName string
	var columns []string
	var values []interface{}

	if matches != nil {
		tableName = matches[1]
		colsStr := matches[2]
		valsStr := matches[3]

		columns = parseColumnList(colsStr)
		values = parseValues(valsStr)
	} else {
		// Try without column names
		re2 := regexp.MustCompile(`(?i)INSERT\s+INTO\s+(\w+)\s+VALUES\s*\((.+)\)`)
		matches = re2.FindStringSubmatch(sql)
		if matches == nil {
			return nil, fmt.Errorf("invalid INSERT: %s", sql)
		}

		tableName = matches[1]
		values = parseValues(matches[2])
	}

	schema, err := e.getSchema(tableName)
	if err != nil {
		return nil, err
	}
	if schema == nil {
		return nil, fmt.Errorf("table '%s' does not exist", tableName)
	}

	// If no columns specified, use schema order
	if columns == nil {
		for _, col := range schema.Columns {
			columns = append(columns, col.Name)
		}
	}

	if len(columns) != len(values) {
		return nil, fmt.Errorf("column count (%d) doesn't match value count (%d)", len(columns), len(values))
	}

	// Create row map
	row := make(map[string]interface{})
	for i, col := range columns {
		row[col] = values[i]
	}

	// Generate row ID
	var rowID string
	if schema.PrimaryKey != "" {
		if val, ok := row[schema.PrimaryKey]; ok {
			rowID = fmt.Sprintf("%v", val)
		} else {
			rowID = uuid.New().String()
		}
	} else {
		rowID = uuid.New().String()
	}

	row["_id"] = rowID

	// Store row
	rowJSON, err := json.Marshal(row)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize row: %w", err)
	}

	if err := e.client.Put(e.rowKey(tableName, rowID), rowJSON); err != nil {
		return nil, fmt.Errorf("failed to store row: %w", err)
	}

	return &SQLQueryResult{
		Rows:         []map[string]interface{}{},
		Columns:      []string{},
		RowsAffected: 1,
	}, nil
}

func (e *SQLEngine) executeSelect(sql string) (*SQLQueryResult, error) {
	// Parse: SELECT cols FROM table [WHERE ...] [ORDER BY ...] [LIMIT ...]
	re := regexp.MustCompile(`(?i)SELECT\s+(.+?)\s+FROM\s+(\w+)`)
	matches := re.FindStringSubmatch(sql)
	if matches == nil {
		return nil, fmt.Errorf("invalid SELECT: %s", sql)
	}

	colsStr := matches[1]
	tableName := matches[2]

	schema, err := e.getSchema(tableName)
	if err != nil {
		return nil, err
	}
	if schema == nil {
		return nil, fmt.Errorf("table '%s' does not exist", tableName)
	}

	// Parse columns
	var columns []string
	if strings.TrimSpace(colsStr) == "*" {
		for _, col := range schema.Columns {
			columns = append(columns, col.Name)
		}
	} else {
		columns = parseColumnList(colsStr)
	}

	// Extract WHERE clause
	var conditions []condition
	whereRe := regexp.MustCompile(`(?i)\bWHERE\s+(.+?)(?:\s+ORDER|\s+LIMIT|\s+OFFSET|$)`)
	if whereMatch := whereRe.FindStringSubmatch(sql); whereMatch != nil {
		conditions = parseWhere(whereMatch[1])
	}

	// Extract ORDER BY
	var orderBy []orderByClause
	orderRe := regexp.MustCompile(`(?i)\bORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s+OFFSET|$)`)
	if orderMatch := orderRe.FindStringSubmatch(sql); orderMatch != nil {
		orderBy = parseOrderBy(orderMatch[1])
	}

	// Extract LIMIT
	var limit int
	limitRe := regexp.MustCompile(`(?i)\bLIMIT\s+(\d+)`)
	if limitMatch := limitRe.FindStringSubmatch(sql); limitMatch != nil {
		limit, _ = strconv.Atoi(limitMatch[1])
	}

	// Extract OFFSET
	var offset int
	offsetRe := regexp.MustCompile(`(?i)\bOFFSET\s+(\d+)`)
	if offsetMatch := offsetRe.FindStringSubmatch(sql); offsetMatch != nil {
		offset, _ = strconv.Atoi(offsetMatch[1])
	}

	// Scan all rows
	prefix := string(e.rowPrefix(tableName))
	results, err := e.client.Scan(prefix)
	if err != nil {
		return nil, fmt.Errorf("failed to scan rows: %w", err)
	}

	var rows []map[string]interface{}
	for _, result := range results {
		var row map[string]interface{}
		if err := json.Unmarshal(result.Value, &row); err != nil {
			continue
		}

		// Apply WHERE conditions
		if matchesConditions(row, conditions) {
			// Project columns
			projected := make(map[string]interface{})
			for _, col := range columns {
				if val, ok := row[col]; ok {
					projected[col] = val
				}
			}
			rows = append(rows, projected)
		}
	}

	// Apply ORDER BY
	if len(orderBy) > 0 {
		sortRows(rows, orderBy)
	}

	// Apply OFFSET
	if offset > 0 && offset < len(rows) {
		rows = rows[offset:]
	} else if offset >= len(rows) {
		rows = []map[string]interface{}{}
	}

	// Apply LIMIT
	if limit > 0 && limit < len(rows) {
		rows = rows[:limit]
	}

	return &SQLQueryResult{
		Rows:         rows,
		Columns:      columns,
		RowsAffected: 0,
	}, nil
}

func (e *SQLEngine) executeUpdate(sql string) (*SQLQueryResult, error) {
	// Parse: UPDATE table SET col1=val1, col2=val2 [WHERE ...]
	re := regexp.MustCompile(`(?i)UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$`)
	matches := re.FindStringSubmatch(sql)
	if matches == nil {
		return nil, fmt.Errorf("invalid UPDATE: %s", sql)
	}

	tableName := matches[1]
	setClause := matches[2]
	whereClause := matches[3]

	schema, err := e.getSchema(tableName)
	if err != nil {
		return nil, err
	}
	if schema == nil {
		return nil, fmt.Errorf("table '%s' does not exist", tableName)
	}

	// Parse SET clause
	updates := parseSetClause(setClause)

	// Parse WHERE clause
	var conditions []condition
	if whereClause != "" {
		conditions = parseWhere(whereClause)
	}

	// Scan all rows
	prefix := string(e.rowPrefix(tableName))
	results, err := e.client.Scan(prefix)
	if err != nil {
		return nil, fmt.Errorf("failed to scan rows: %w", err)
	}

	rowsAffected := 0
	for _, result := range results {
		var row map[string]interface{}
		if err := json.Unmarshal(result.Value, &row); err != nil {
			continue
		}

		// Apply WHERE conditions
		if matchesConditions(row, conditions) {
			// Apply updates
			for col, val := range updates {
				row[col] = val
			}

			// Save updated row
			rowJSON, err := json.Marshal(row)
			if err != nil {
				continue
			}

			if err := e.client.Put(result.Key, rowJSON); err != nil {
				continue
			}

			rowsAffected++
		}
	}

	return &SQLQueryResult{
		Rows:         []map[string]interface{}{},
		Columns:      []string{},
		RowsAffected: rowsAffected,
	}, nil
}

func (e *SQLEngine) executeDelete(sql string) (*SQLQueryResult, error) {
	// Parse: DELETE FROM table [WHERE ...]
	re := regexp.MustCompile(`(?i)DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?$`)
	matches := re.FindStringSubmatch(sql)
	if matches == nil {
		return nil, fmt.Errorf("invalid DELETE: %s", sql)
	}

	tableName := matches[1]
	whereClause := matches[2]

	schema, err := e.getSchema(tableName)
	if err != nil {
		return nil, err
	}
	if schema == nil {
		return nil, fmt.Errorf("table '%s' does not exist", tableName)
	}

	// Parse WHERE clause
	var conditions []condition
	if whereClause != "" {
		conditions = parseWhere(whereClause)
	}

	// Scan all rows
	prefix := string(e.rowPrefix(tableName))
	results, err := e.client.Scan(prefix)
	if err != nil {
		return nil, fmt.Errorf("failed to scan rows: %w", err)
	}

	rowsAffected := 0
	for _, result := range results {
		var row map[string]interface{}
		if err := json.Unmarshal(result.Value, &row); err != nil {
			continue
		}

		// Apply WHERE conditions
		if matchesConditions(row, conditions) {
			if err := e.client.Delete(result.Key); err != nil {
				continue
			}
			rowsAffected++
		}
	}

	return &SQLQueryResult{
		Rows:         []map[string]interface{}{},
		Columns:      []string{},
		RowsAffected: rowsAffected,
	}, nil
}

// Helper functions

type condition struct {
	column   string
	operator string
	value    interface{}
}

type orderByClause struct {
	column    string
	direction string
}

func parseColumns(colsStr string) ([]Column, string) {
	var columns []Column
	var primaryKey string

	parts := splitByComma(colsStr)

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		upperPart := strings.ToUpper(part)

		// Check for PRIMARY KEY constraint
		if strings.HasPrefix(upperPart, "PRIMARY KEY") {
			re := regexp.MustCompile(`(?i)PRIMARY\s+KEY\s*\((\w+)\)`)
			if match := re.FindStringSubmatch(part); match != nil {
				primaryKey = match[1]
			}
			continue
		}

		// Parse column definition
		tokens := strings.Fields(part)
		if len(tokens) < 2 {
			continue
		}

		colName := tokens[0]
		colType := strings.ToUpper(tokens[1])

		// Normalize types
		switch colType {
		case "INTEGER", "INT", "BIGINT", "SMALLINT":
			colType = "INT"
		case "VARCHAR", "CHAR", "STRING", "TEXT":
			colType = "TEXT"
		case "REAL", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC":
			colType = "FLOAT"
		case "BOOLEAN", "BOOL":
			colType = "BOOL"
		case "BLOB", "BYTES", "BINARY":
			colType = "BLOB"
		}

		isPK := strings.Contains(upperPart, "PRIMARY KEY")
		nullable := !strings.Contains(upperPart, "NOT NULL")

		if isPK {
			primaryKey = colName
		}

		columns = append(columns, Column{
			Name:       colName,
			Type:       colType,
			Nullable:   nullable,
			PrimaryKey: isPK,
		})
	}

	return columns, primaryKey
}

func parseColumnList(colsStr string) []string {
	parts := splitByComma(colsStr)
	var columns []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			columns = append(columns, part)
		}
	}
	return columns
}

func parseValues(valsStr string) []interface{} {
	var values []interface{}
	var current strings.Builder
	inString := false
	var stringChar rune

	for _, char := range valsStr {
		if (char == '"' || char == '\'') && !inString {
			inString = true
			stringChar = char
			current.WriteRune(char)
		} else if char == stringChar && inString {
			inString = false
			stringChar = 0
			current.WriteRune(char)
		} else if char == ',' && !inString {
			values = append(values, parseValue(strings.TrimSpace(current.String())))
			current.Reset()
		} else {
			current.WriteRune(char)
		}
	}

	if current.Len() > 0 {
		values = append(values, parseValue(strings.TrimSpace(current.String())))
	}

	return values
}

func parseValue(valStr string) interface{} {
	valStr = strings.TrimSpace(valStr)

	if valStr == "" || strings.ToUpper(valStr) == "NULL" {
		return nil
	}

	// String literals
	if (strings.HasPrefix(valStr, "'") && strings.HasSuffix(valStr, "'")) ||
		(strings.HasPrefix(valStr, "\"") && strings.HasSuffix(valStr, "\"")) {
		return valStr[1 : len(valStr)-1]
	}

	// Boolean
	upper := strings.ToUpper(valStr)
	if upper == "TRUE" {
		return true
	}
	if upper == "FALSE" {
		return false
	}

	// Numbers
	if strings.Contains(valStr, ".") {
		if f, err := strconv.ParseFloat(valStr, 64); err == nil {
			return f
		}
	} else {
		if i, err := strconv.Atoi(valStr); err == nil {
			return i
		}
	}

	return valStr
}

func parseWhere(whereClause string) []condition {
	var conditions []condition

	// Split by AND
	parts := regexp.MustCompile(`(?i)\s+AND\s+`).Split(whereClause, -1)

	for _, part := range parts {
		part = strings.TrimSpace(part)

		// Match: column operator value
		re := regexp.MustCompile(`(\w+)\s*(=|!=|<>|>=|<=|>|<|LIKE|NOT\s+LIKE)\s*(.+)`)
		if match := re.FindStringSubmatch(part); match != nil {
			col := match[1]
			op := strings.ToUpper(strings.ReplaceAll(match[2], " ", "_"))
			if op == "<>" {
				op = "!="
			}
			val := parseValue(match[3])

			conditions = append(conditions, condition{
				column:   col,
				operator: op,
				value:    val,
			})
		}
	}

	return conditions
}

func parseOrderBy(orderByStr string) []orderByClause {
	var orderBy []orderByClause

	parts := splitByComma(orderByStr)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		upper := strings.ToUpper(part)

		var col, dir string
		if strings.HasSuffix(upper, " DESC") {
			col = strings.TrimSpace(part[:len(part)-5])
			dir = "DESC"
		} else if strings.HasSuffix(upper, " ASC") {
			col = strings.TrimSpace(part[:len(part)-4])
			dir = "ASC"
		} else {
			col = part
			dir = "ASC"
		}

		orderBy = append(orderBy, orderByClause{column: col, direction: dir})
	}

	return orderBy
}

func parseSetClause(setClause string) map[string]interface{} {
	updates := make(map[string]interface{})

	parts := splitByComma(setClause)
	for _, part := range parts {
		if eqIdx := strings.Index(part, "="); eqIdx > 0 {
			col := strings.TrimSpace(part[:eqIdx])
			val := parseValue(strings.TrimSpace(part[eqIdx+1:]))
			updates[col] = val
		}
	}

	return updates
}

func matchesConditions(row map[string]interface{}, conditions []condition) bool {
	for _, cond := range conditions {
		rowVal := row[cond.column]

		switch cond.operator {
		case "=":
			if !compareValues(rowVal, cond.value, "=") {
				return false
			}
		case "!=":
			if compareValues(rowVal, cond.value, "=") {
				return false
			}
		case ">":
			if !compareValues(rowVal, cond.value, ">") {
				return false
			}
		case ">=":
			if !compareValues(rowVal, cond.value, ">=") {
				return false
			}
		case "<":
			if !compareValues(rowVal, cond.value, "<") {
				return false
			}
		case "<=":
			if !compareValues(rowVal, cond.value, "<=") {
				return false
			}
		case "LIKE":
			if !matchLike(rowVal, cond.value) {
				return false
			}
		case "NOT_LIKE":
			if matchLike(rowVal, cond.value) {
				return false
			}
		}
	}

	return true
}

func compareValues(a, b interface{}, op string) bool {
	if a == nil || b == nil {
		return op == "=" && a == b
	}

	// Convert to comparable types
	af := toFloat64(a)
	bf := toFloat64(b)

	switch op {
	case "=":
		return af == bf
	case ">":
		return af > bf
	case ">=":
		return af >= bf
	case "<":
		return af < bf
	case "<=":
		return af <= bf
	}

	return false
}

func toFloat64(v interface{}) float64 {
	switch val := v.(type) {
	case int:
		return float64(val)
	case int64:
		return float64(val)
	case float64:
		return val
	case float32:
		return float64(val)
	case string:
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			return f
		}
	}
	return 0
}

func matchLike(rowVal, pattern interface{}) bool {
	if rowVal == nil {
		return false
	}

	rowStr := fmt.Sprintf("%v", rowVal)
	patternStr := fmt.Sprintf("%v", pattern)

	// Convert SQL LIKE to regex
	patternStr = regexp.QuoteMeta(patternStr)
	patternStr = strings.ReplaceAll(patternStr, "%", ".*")
	patternStr = strings.ReplaceAll(patternStr, "_", ".")

	re := regexp.MustCompile("(?i)^" + patternStr + "$")
	return re.MatchString(rowStr)
}

func sortRows(rows []map[string]interface{}, orderBy []orderByClause) {
	// Simple bubble sort for ORDER BY
	for i := 0; i < len(rows)-1; i++ {
		for j := 0; j < len(rows)-i-1; j++ {
			shouldSwap := false

			for _, order := range orderBy {
				val1 := rows[j][order.column]
				val2 := rows[j+1][order.column]

				cmp := compareForSort(val1, val2)
				if cmp == 0 {
					continue
				}

				if order.direction == "DESC" {
					shouldSwap = cmp < 0
				} else {
					shouldSwap = cmp > 0
				}
				break
			}

			if shouldSwap {
				rows[j], rows[j+1] = rows[j+1], rows[j]
			}
		}
	}
}

func compareForSort(a, b interface{}) int {
	if a == nil && b == nil {
		return 0
	}
	if a == nil {
		return 1
	}
	if b == nil {
		return -1
	}

	af := toFloat64(a)
	bf := toFloat64(b)

	if af < bf {
		return -1
	} else if af > bf {
		return 1
	}
	return 0
}

func splitByComma(s string) []string {
	var result []string
	var current strings.Builder
	depth := 0
	inString := false
	var stringChar rune

	for _, char := range s {
		if (char == '"' || char == '\'') && !inString {
			inString = true
			stringChar = char
			current.WriteRune(char)
		} else if char == stringChar && inString {
			inString = false
			stringChar = 0
			current.WriteRune(char)
		} else if char == '(' && !inString {
			depth++
			current.WriteRune(char)
		} else if char == ')' && !inString {
			depth--
			current.WriteRune(char)
		} else if char == ',' && depth == 0 && !inString {
			result = append(result, current.String())
			current.Reset()
		} else {
			current.WriteRune(char)
		}
	}

	if current.Len() > 0 {
		result = append(result, current.String())
	}

	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

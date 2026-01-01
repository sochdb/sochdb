// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! SQL-specific error types

use thiserror::Error;

/// SQL execution errors
#[derive(Error, Debug, Clone)]
pub enum SqlError {
    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("Lexer error: {0}")]
    LexError(String),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Type error: {0}")]
    TypeError(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Transaction error: {0}")]
    TransactionError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl SqlError {
    pub fn from_parse_errors(errors: Vec<super::parser::ParseError>) -> Self {
        if let Some(first) = errors.first() {
            SqlError::ParseError {
                message: first.message.clone(),
                line: first.span.line,
                column: first.span.column,
            }
        } else {
            SqlError::ParseError {
                message: "Unknown parse error".to_string(),
                line: 0,
                column: 0,
            }
        }
    }
}

pub type SqlResult<T> = std::result::Result<T, SqlError>;

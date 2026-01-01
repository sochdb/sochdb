//! ToonQL parser for ToonDB-specific query syntax

use serde::{Deserialize, Serialize};

/// ToonQL statement type
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToonQlStatement {
    /// VECTOR_SEARCH table USING column NEAR query LIMIT n
    VectorSearch {
        table: String,
        column: String,
        query: VectorQuery,
        limit: usize,
        metric: Option<String>,
    },
    /// GET 'path/to/key'
    Get { path: String },
    /// PUT 'path/to/key' = value
    Put { path: String, value: String },
    /// DELETE 'path/to/key'
    Delete { path: String },
    /// SCAN 'prefix/*'
    Scan { prefix: String },
    /// Unknown/unparsed statement
    Unknown(String),
}

/// Vector query type
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorQuery {
    /// Text to be embedded
    Text(String),
    /// Raw vector values
    Vector(Vec<f32>),
}

/// Parse a ToonQL query string
#[allow(dead_code)]
pub fn parse_toonql(query: &str) -> Result<ToonQlStatement, String> {
    let query = query.trim();
    let upper = query.to_uppercase();

    if upper.starts_with("VECTOR_SEARCH") {
        parse_vector_search(query)
    } else if upper.starts_with("GET ") {
        parse_get(query)
    } else if upper.starts_with("PUT ") {
        parse_put(query)
    } else if upper.starts_with("DELETE ") {
        parse_delete(query)
    } else if upper.starts_with("SCAN ") {
        parse_scan(query)
    } else {
        Ok(ToonQlStatement::Unknown(query.to_string()))
    }
}

#[allow(dead_code)]
fn parse_vector_search(input: &str) -> Result<ToonQlStatement, String> {
    let upper = input.to_uppercase();
    
    // Extract table name (after VECTOR_SEARCH)
    let after_vs = &input[13..].trim();
    let table_end = after_vs.find(|c: char| c.is_whitespace()).unwrap_or(after_vs.len());
    let table = after_vs[..table_end].to_string();

    // Extract USING column
    let using_idx = upper.find(" USING ").ok_or("Missing USING clause")?;
    let after_using = &input[using_idx + 7..];
    let column_end = after_using.find(|c: char| c.is_whitespace()).unwrap_or(after_using.len());
    let column = after_using[..column_end].trim().to_string();

    // Extract NEAR query
    let near_idx = upper.find(" NEAR ").ok_or("Missing NEAR clause")?;
    let after_near = &input[near_idx + 6..];
    
    // Parse query (text or vector)
    let vector_query = if after_near.starts_with('[') {
        // Vector format
        let end = after_near.find(']').unwrap_or(after_near.len());
        let vec_str = &after_near[1..end];
        let values: Result<Vec<f32>, _> = vec_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();
        VectorQuery::Vector(values.map_err(|e| e.to_string())?)
    } else if after_near.starts_with('\'') {
        // Text format
        let end = after_near[1..].find('\'').unwrap_or(after_near.len() - 1);
        VectorQuery::Text(after_near[1..=end].to_string())
    } else {
        return Err("Invalid NEAR query format".to_string());
    };

    // Extract LIMIT
    let limit = if let Some(limit_idx) = upper.find(" LIMIT ") {
        let limit_str = &input[limit_idx + 7..];
        let end = limit_str.find(|c: char| !c.is_numeric()).unwrap_or(limit_str.len());
        limit_str[..end].trim().parse().unwrap_or(10)
    } else {
        10
    };

    Ok(ToonQlStatement::VectorSearch {
        table,
        column,
        query: vector_query,
        limit,
        metric: None,
    })
}

#[allow(dead_code)]
fn parse_get(query: &str) -> Result<ToonQlStatement, String> {
    let after_get = &query[4..].trim();
    let path = extract_quoted_string(after_get)?;
    Ok(ToonQlStatement::Get { path })
}

#[allow(dead_code)]
fn parse_put(query: &str) -> Result<ToonQlStatement, String> {
    let after_put = &query[4..].trim();
    
    // Find the '=' separator
    let eq_idx = after_put.find('=').ok_or("Missing '=' in PUT statement")?;
    
    let path = extract_quoted_string(&after_put[..eq_idx].trim())?;
    let value = after_put[eq_idx + 1..].trim().to_string();
    
    Ok(ToonQlStatement::Put { path, value })
}

#[allow(dead_code)]
fn parse_delete(query: &str) -> Result<ToonQlStatement, String> {
    let after_delete = &query[7..].trim();
    let path = extract_quoted_string(after_delete)?;
    Ok(ToonQlStatement::Delete { path })
}

#[allow(dead_code)]
fn parse_scan(query: &str) -> Result<ToonQlStatement, String> {
    let after_scan = &query[5..].trim();
    let prefix = extract_quoted_string(after_scan)?;
    Ok(ToonQlStatement::Scan { prefix })
}

#[allow(dead_code)]
fn extract_quoted_string(s: &str) -> Result<String, String> {
    let s = s.trim();
    if s.starts_with('\'') && s.len() > 1 {
        let end = s[1..].find('\'').unwrap_or(s.len() - 1);
        Ok(s[1..=end].to_string())
    } else {
        Err("Expected quoted string".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_get() {
        let result = parse_toonql("GET 'config/settings/theme'");
        assert!(result.is_ok());
        
        if let ToonQlStatement::Get { path } = result.unwrap() {
            assert_eq!(path, "config/settings/theme");
        } else {
            panic!("Expected GET statement");
        }
    }

    #[test]
    fn test_parse_vector_search() {
        let result = parse_toonql("VECTOR_SEARCH documents USING embedding NEAR 'machine learning' LIMIT 10");
        assert!(result.is_ok());
        
        if let ToonQlStatement::VectorSearch { table, column, limit, .. } = result.unwrap() {
            assert_eq!(table, "documents");
            assert_eq!(column, "embedding");
            assert_eq!(limit, 10);
        } else {
            panic!("Expected VECTOR_SEARCH statement");
        }
    }
}

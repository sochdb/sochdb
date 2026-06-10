//! Canonical SQL `LIKE` pattern matching.
//!
//! This is the single source of truth for `LIKE` semantics across every query
//! path (storage scan, filter pushdown, expression evaluation, virtual/plugin
//! tables). Historically the engine carried four divergent implementations that
//! disagreed on case sensitivity and on whether regex metacharacters in the
//! subject/pattern were treated literally. They have all been routed through
//! [`like_match`] so that `LIKE` behaves identically regardless of the path a
//! query takes.
//!
//! # Semantics
//! - Case **sensitive** (SQL-92 standard; collation-aware folding is out of scope).
//! - `%` matches zero or more characters.
//! - `_` matches exactly one character.
//! - Every other character — including regex metacharacters such as `.`, `(`,
//!   `*`, `[` — matches itself literally. The matcher does not use a regex
//!   engine, so no escaping pitfalls exist.

/// Returns `true` if `s` matches the SQL `LIKE` `pattern`.
///
/// `%` matches any run of characters (including empty), `_` matches exactly one
/// character, and all other characters are matched literally and
/// case-sensitively.
pub fn like_match(s: &str, pattern: &str) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    let p_chars: Vec<char> = pattern.chars().collect();
    like_match_chars(&s_chars, &p_chars)
}

/// Iterative LIKE matcher with greedy `%` backtracking.
///
/// Runs in O(len(s) * len(pattern)) time and O(1) extra space by tracking the
/// most recent `%` position to backtrack to, avoiding the exponential blowup of
/// naive recursive backtracking on patterns with many `%` wildcards.
fn like_match_chars(s: &[char], p: &[char]) -> bool {
    let (mut si, mut pi) = (0usize, 0usize);
    // Backtrack anchors: where to resume if a tentative `%` match fails.
    let mut star_pi: Option<usize> = None;
    let mut star_si = 0usize;

    while si < s.len() {
        if pi < p.len() && p[pi] == '%' {
            // Record the `%` and assume it matches the empty string for now.
            star_pi = Some(pi);
            star_si = si;
            pi += 1;
        } else if pi < p.len() && (p[pi] == '_' || p[pi] == s[si]) {
            si += 1;
            pi += 1;
        } else if let Some(spi) = star_pi {
            // Mismatch: let the last `%` consume one more character.
            star_si += 1;
            si = star_si;
            pi = spi + 1;
        } else {
            return false;
        }
    }

    // Consume any trailing `%` wildcards (they can match the empty string).
    while pi < p.len() && p[pi] == '%' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::like_match;

    #[test]
    fn test_basic_wildcards() {
        assert!(like_match("hello", "hello"));
        assert!(like_match("hello", "%llo"));
        assert!(like_match("hello", "h%o"));
        assert!(like_match("hello", "h_llo"));
        assert!(like_match("hello", "%"));
        assert!(like_match("", "%"));
        assert!(like_match("hello", "h%l%o"));
        assert!(!like_match("hello", "world"));
        assert!(!like_match("hello", "hell"));
        assert!(!like_match("hello", "_ello_"));
    }

    #[test]
    fn test_literal_metacharacters() {
        // Regex metacharacters must be matched literally, not as regex.
        assert!(like_match("file.txt", "file%"));
        assert!(like_match("file.txt", "file.txt"));
        assert!(!like_match("fileXtxt", "file.txt")); // '.' is literal, not "any char"
        assert!(like_match("test(1)", "%(%"));
        assert!(like_match("a+b", "a+b"));
        assert!(like_match("a[b]", "a[b]"));
    }

    #[test]
    fn test_case_sensitive() {
        assert!(like_match("Hello", "Hello"));
        assert!(!like_match("Hello", "hello"));
        assert!(!like_match("HELLO", "%llo"));
    }

    #[test]
    fn test_many_percent_no_blowup() {
        // Pathological pattern that would blow up under naive recursion.
        let s = "a".repeat(50);
        assert!(like_match(&s, "%a%a%a%a%a%a%a%a%a%a%"));
        let s2 = "a".repeat(49) + "b";
        assert!(!like_match(&s2, "%a%a%a%a%a%a%a%a%a%a%a%c"));
    }
}

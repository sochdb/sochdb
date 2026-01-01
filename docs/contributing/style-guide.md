# Documentation Style Guide

Guidelines for writing ToonDB documentation.

---

## Principles

1. **Clarity over cleverness** — Write for understanding, not impression
2. **Scannable** — Use headers, lists, and tables
3. **Actionable** — Include working code examples
4. **Current** — Keep docs in sync with code

---

## Structure

### Page Template

```markdown
# Page Title

Brief description of what this page covers.

---

## Table of Contents (optional, for long pages)

---

## First Section

Content...

### Subsection

More content...

---

## See Also

- [Related Page](./related.md)
```

### Diátaxis Quadrants

Place each document in the appropriate quadrant:

| Type | Purpose | Location |
|------|---------|----------|
| **Tutorial** | Learning-oriented | `docs/tutorials/` |
| **How-to** | Problem-oriented | `docs/cookbook/` |
| **Reference** | Information-oriented | `docs/reference/` or `docs/API.md` |
| **Explanation** | Understanding-oriented | `docs/internals/` or `docs/explanation/` |

---

## Writing Style

### Voice

- Use **second person** ("you can...") for instructions
- Use **active voice** ("ToonDB stores..." not "data is stored...")
- Be **direct** ("Run this command" not "You might want to run...")

### Headings

- Use **sentence case** (capitalize first word only)
- Make headings **descriptive** ("How to configure logging" not "Logging")
- Limit to **3 levels** (H2, H3, H4)

### Code Examples

Always include:
1. **Language identifier** in code fences
2. **Runnable examples** when possible
3. **Expected output** for commands

```python
# ✅ Good: Complete, runnable example
from toondb import Database

db = Database.open("./my_db")
db.put(b"key", b"value")
print(db.get(b"key"))  # Output: b"value"
db.close()
```

```python
# ❌ Bad: Incomplete, can't run
db.put("key", "value")  # Missing import, wrong type
```

### Lists

- Use **bullet lists** for unordered items
- Use **numbered lists** for sequential steps
- Use **tables** for structured data with multiple attributes

### Links

- Use **relative paths** for internal links
- Include **descriptive text** (not "click here")
- Link to **specific sections** when helpful

```markdown
# ✅ Good
See the [configuration reference](./reference/configuration.md#storage-options)

# ❌ Bad
Click [here](./reference/configuration.md) for more info
```

---

## Formatting

### Code

- **Inline code** for: file names, function names, commands, paths
  - `toondb.toml`, `ToonValue`, `cargo test`, `/var/lib/toondb`
  
- **Code blocks** for: multi-line code, command output, file contents

### Tables

Use tables for comparison and reference:

```markdown
| Feature | Free | Pro |
|---------|------|-----|
| Storage | 1 GB | 100 GB |
| Support | Community | Priority |
```

### Callouts

Use blockquotes for important notes:

```markdown
> **Note:** This is important information.

> **Warning:** This action cannot be undone.

> **Tip:** This makes things easier.
```

---

## Content Guidelines

### Tutorials

- Start with a clear **goal**
- List **prerequisites**
- Provide **step-by-step** instructions
- Include **verification steps**
- End with **next steps**

### How-To Guides

- Start with the **problem**
- Provide the **solution** immediately
- Add **examples**
- Discuss **alternatives/trade-offs**
- Link to **related recipes**

### Reference

- Be **complete** (document all options)
- Use **consistent format** for each entry
- Include **types and defaults**
- Provide **examples** for complex options

### Explanation

- Start with **why** before **how**
- Use **diagrams** for architecture
- Explain **trade-offs** in decisions
- Link to **source code** when relevant

---

## Diagrams

Use ASCII diagrams for architecture:

```
┌─────────┐     ┌─────────┐
│ Client  │────▶│ Server  │
└─────────┘     └─────────┘
```

For complex diagrams, use Mermaid (if supported) or link to images.

---

## Versioning

### API Changes

Mark deprecated features:

```markdown
> **Deprecated in v0.2:** Use `new_method()` instead.
```

### Version-Specific Content

```markdown
> **New in v0.2:** This feature was added in version 0.2.
```

---

## Review Checklist

Before submitting documentation:

- [ ] Code examples compile/run
- [ ] Links work (no 404s)
- [ ] Spelling and grammar checked
- [ ] Follows style guide
- [ ] Placed in correct Diátaxis quadrant
- [ ] Added to navigation/index

---

## Tools

### Spell Check

```bash
# Using aspell
find docs -name "*.md" -exec aspell check {} \;
```

### Link Check

```bash
# Using markdown-link-check
npx markdown-link-check docs/**/*.md
```

### Preview

```bash
# Using grip (GitHub-flavored markdown)
pip install grip
grip docs/index.md
```

---

## See Also

- [Testing Guide](/contributing/testing) — How to test
- [Diátaxis Framework](https://diataxis.fr/) — Documentation methodology
- [Google Developer Documentation Style Guide](https://developers.google.com/style)


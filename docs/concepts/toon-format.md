# TOON Format Specification

> **Tabular Object-Oriented Notation** — Token-efficient data format for LLMs

---

## Overview

TOON (Tabular Object-Oriented Notation) is a compact data serialization format designed to minimize tokens when data is consumed by LLMs.

### Token Comparison

| Format | 100 rows × 5 fields | Token Count |
|--------|---------------------|-------------|
| JSON | Full object notation | ~7,500 tokens |
| CSV | Headers + rows | ~4,200 tokens |
| **TOON** | Compact notation | **~2,550 tokens** |

**TOON achieves 40-66% token reduction** compared to JSON.

---

## Text Format

### Basic Syntax

```
tablename[rowcount]{field1,field2,...}:
value1,value2,...;
value1,value2,...;
```

### Examples

**Simple table:**
```
users[3]{id,name,email}:
1,Alice,alice@example.com;
2,Bob,bob@example.com;
3,Carol,carol@example.com
```

**With types:**
```
orders[2]{id:uint,amount:float,status:text}:
1001,99.99,pending;
1002,149.50,shipped
```

**Nested/complex:**
```
products[2]{id,name,tags,metadata}:
1,Widget,[red,blue],{color:red,size:M};
2,Gadget,[green],{color:green,size:L}
```

---

## Grammar (EBNF)

```ebnf
document     ::= table_header row_data
table_header ::= name "[" count "]" "{" fields "}" ":"
name         ::= identifier
count        ::= integer
fields       ::= field ("," field)*
field        ::= identifier (":" type)?
type         ::= "int" | "uint" | "float" | "text" | "bool" | "bytes" 
               | "vec(" integer ")" | type "?"

row_data     ::= row (";" row)* ";"?
row          ::= value ("," value)*

value        ::= null | bool | number | string | array | object | ref
null         ::= "∅" | "null"
bool         ::= "T" | "F" | "true" | "false"
number       ::= integer | float
integer      ::= "-"? digit+
float        ::= "-"? digit+ "." digit+
string       ::= raw_string | quoted_string
raw_string   ::= [^,;\n"{}[\]]+
quoted_string::= '"' ([^"\\] | escape)* '"'
array        ::= "[" (value ("," value)*)? "]"
object       ::= "{" (pair ("," pair)*)? "}"
pair         ::= identifier ":" value
ref          ::= "ref(" identifier "," integer ")"
escape       ::= "\\" ["\\/bfnrt]
```

---

## Type System

### Primitive Types

| Type | Example | Description |
|------|---------|-------------|
| `int` | `42`, `-17` | Signed 64-bit integer |
| `uint` | `42` | Unsigned 64-bit integer |
| `float` | `3.14`, `-0.5` | 64-bit floating point |
| `text` | `hello`, `"with spaces"` | UTF-8 string |
| `bool` | `T`, `F` | Boolean |
| `bytes` | `<base64>` | Binary data |
| `null` | `∅` | Null value |

### Complex Types

| Type | Example | Description |
|------|---------|-------------|
| `vec(N)` | `[0.1,0.2,0.3]` | N-dimensional vector |
| `array` | `[1,2,3]` | Heterogeneous array |
| `object` | `{key:value}` | Key-value object |
| `ref(T,id)` | `ref(users,42)` | Reference to another table |

### Optional Types

Append `?` to make nullable:

```
users[2]{id:uint,email:text,phone:text?}:
1,alice@ex.com,555-0100;
2,bob@ex.com,∅
```

---

## Encoding Rules

### Strings

- **Unquoted**: If string contains no special characters
  ```
  hello
  simple_name
  user@example.com
  ```

- **Quoted**: If string contains `,`, `;`, `"`, `{`, `}`, `[`, `]`, or newlines
  ```
  "Hello, World"
  "Line 1\nLine 2"
  "Say \"Hello\""
  ```

### Numbers

- Integers: No decimal point
- Floats: Always include decimal point
- Scientific notation: `1.5e10`

### Nulls

- `∅` (Unicode null symbol, recommended)
- `null` (word form)

### Booleans

- `T` / `F` (short form, recommended)
- `true` / `false` (word form)

### Arrays

Square brackets, comma-separated:
```
[1,2,3]
[red,green,blue]
[[1,2],[3,4]]
```

### Objects

Curly braces, colon-separated key-value pairs:
```
{name:Alice,age:30}
{nested:{inner:value}}
```

### References

Foreign key references to other tables:
```
ref(users,42)
ref(orders,1001)
```

---

## Binary Format

For internal storage and high-performance scenarios, TOON has a binary encoding.

### Header (16 bytes)

```
┌──────────┬──────────┬──────────┬──────────┐
│  Magic   │ Version  │  Flags   │ Row Count│
│ "TOON"   │ u16      │ u16      │ u64      │
│ 4 bytes  │ 2 bytes  │ 2 bytes  │ 8 bytes  │
└──────────┴──────────┴──────────┴──────────┘
```

### Type Tags

```rust
enum ToonTypeTag {
    Null      = 0x00,
    False     = 0x01,
    True      = 0x02,
    
    // Fixed-size integers (value in tag)
    PosFixint = 0x10,  // 0-15 in lower nibble
    NegFixint = 0x20,  // -16 to -1
    
    // Variable-size integers
    Int8      = 0x30,
    Int16     = 0x31,
    Int32     = 0x32,
    Int64     = 0x33,
    
    // Floats
    Float32   = 0x40,
    Float64   = 0x41,
    
    // Strings
    FixStr    = 0x50,  // 0-15 length in lower nibble
    Str8      = 0x60,  // 1-byte length prefix
    Str16     = 0x61,  // 2-byte length prefix
    Str32     = 0x62,  // 4-byte length prefix
    
    // Complex
    Array     = 0x70,
    Object    = 0x71,
    Ref       = 0x80,
    Vector    = 0x90,
}
```

### Varint Encoding

Large integers use variable-length encoding:

```
Value Range          | Bytes
---------------------|-------
0-127                | 1
128-16383            | 2
16384-2097151        | 3
...                  | ...
```

---

## Parsing

### Rust

```rust
use toondb_core::ToonCodec;

// Parse text TOON
let text = r#"users[2]{id,name}:1,Alice;2,Bob"#;
let table = ToonCodec::parse_text(text)?;

// Parse binary TOON
let binary = read_file("data.toon")?;
let table = ToonCodec::parse_binary(&binary)?;
```

### Python

```python
from toondb import parse_toon

text = "users[2]{id,name}:1,Alice;2,Bob"
table = parse_toon(text)

for row in table.rows:
    print(row["id"], row["name"])
```

---

## Encoding

### Rust

```rust
use toondb_core::{ToonTable, ToonCodec};

let table = ToonTable::new("users")
    .field("id", ToonType::UInt)
    .field("name", ToonType::Text)
    .row(vec![1.into(), "Alice".into()])
    .row(vec![2.into(), "Bob".into()]);

// To text
let text = ToonCodec::encode_text(&table);
// "users[2]{id,name}:1,Alice;2,Bob"

// To binary
let binary = ToonCodec::encode_binary(&table);
```

### Python

```python
from toondb import ToonTable, encode_toon

table = ToonTable("users", ["id", "name"])
table.add_row([1, "Alice"])
table.add_row([2, "Bob"])

text = encode_toon(table)
# "users[2]{id,name}:1,Alice;2,Bob"
```

---

## Best Practices

### For LLMs

1. **Always use TOON** for tabular data in prompts
2. **Short field names** reduce tokens further
3. **Omit types** when inferable from context
4. **Use ∅** instead of `null` (fewer tokens)

### For Storage

1. **Binary format** for internal storage
2. **Text format** for debugging and logging
3. **Compress** binary TOON with LZ4 or ZSTD

### Token Optimization Tips

```
# More tokens (explicit types)
users[2]{id:uint,name:text,email:text}:1,Alice,alice@ex.com;2,Bob,bob@ex.com

# Fewer tokens (inferred types)
users[2]{id,name,email}:1,Alice,alice@ex.com;2,Bob,bob@ex.com

# Even fewer (short names)
u[2]{i,n,e}:1,Alice,alice@ex.com;2,Bob,bob@ex.com
```

---

## See Also

- [API Reference](/api-reference/python-api) — ToonCodec API
- [Architecture](/concepts/architecture) — Internal format details
- [Performance](/concepts/performance) — Optimization benchmarks


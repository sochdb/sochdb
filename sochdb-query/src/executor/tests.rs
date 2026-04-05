// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tests for the Volcano executor module.

#[cfg(test)]
mod tests {
    use crate::executor::types::{Row, Schema, ColumnMeta};
    use crate::executor::node::PlanNode;
    use crate::executor::scan::{ValuesNode, EmptyNode};
    use crate::executor::filter::FilterNode;
    use crate::executor::project::{ProjectNode, ProjectExpr, PassThroughNode};
    use crate::executor::sort::{SortNode, SortKey};
    use crate::executor::limit::LimitNode;
    use crate::executor::aggregate::{HashAggregateNode, AggFunc, AggDef};
    use crate::executor::join::HashJoinNode;
    use crate::soch_ql::SochValue;
    use crate::sql::ast::{Expr, Literal, ColumnRef, BinaryOperator, JoinType};

    // ========================================================================
    // Helpers
    // ========================================================================

    fn schema_abc() -> Schema {
        Schema::new(vec![
            ColumnMeta::new("a"),
            ColumnMeta::new("b"),
            ColumnMeta::new("c"),
        ])
    }

    fn schema_id_name_age() -> Schema {
        Schema::new(vec![
            ColumnMeta::new("id"),
            ColumnMeta::new("name"),
            ColumnMeta::new("age"),
        ])
    }

    fn sample_rows() -> Vec<Row> {
        vec![
            vec![SochValue::Int(1), SochValue::Text("Alice".into()), SochValue::Int(30)],
            vec![SochValue::Int(2), SochValue::Text("Bob".into()), SochValue::Int(25)],
            vec![SochValue::Int(3), SochValue::Text("Carol".into()), SochValue::Int(35)],
            vec![SochValue::Int(4), SochValue::Text("Dave".into()), SochValue::Int(25)],
            vec![SochValue::Int(5), SochValue::Text("Eve".into()), SochValue::Int(30)],
        ]
    }

    fn values_node(schema: Schema, rows: Vec<Row>) -> Box<dyn PlanNode> {
        Box::new(ValuesNode::new(schema, rows))
    }

    // ========================================================================
    // Schema tests
    // ========================================================================

    #[test]
    fn test_schema_index_of() {
        let s = schema_abc();
        assert_eq!(s.index_of("a"), Some(0));
        assert_eq!(s.index_of("b"), Some(1));
        assert_eq!(s.index_of("c"), Some(2));
        assert_eq!(s.index_of("d"), None);
    }

    #[test]
    fn test_schema_qualified_index() {
        let s = Schema::new(vec![
            ColumnMeta::qualified("t1", "id"),
            ColumnMeta::qualified("t2", "id"),
        ]);
        assert_eq!(s.index_of_qualified(Some("t1"), "id"), Some(0));
        assert_eq!(s.index_of_qualified(Some("t2"), "id"), Some(1));
        assert_eq!(s.index_of_qualified(None, "id"), Some(0)); // First match
    }

    #[test]
    fn test_schema_merge() {
        let s1 = Schema::new(vec![ColumnMeta::new("a")]);
        let s2 = Schema::new(vec![ColumnMeta::new("b")]);
        let merged = s1.merge(&s2);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.column_names(), vec!["a", "b"]);
    }

    // ========================================================================
    // ValuesNode tests
    // ========================================================================

    #[test]
    fn test_values_node_basic() {
        let rows = vec![
            vec![SochValue::Int(1), SochValue::Text("hello".into())],
            vec![SochValue::Int(2), SochValue::Text("world".into())],
        ];
        let schema = Schema::new(vec![ColumnMeta::new("x"), ColumnMeta::new("y")]);
        let mut node = ValuesNode::new(schema, rows);

        let r1 = node.next().unwrap();
        assert!(r1.is_some());
        assert_eq!(r1.unwrap()[0], SochValue::Int(1));

        let r2 = node.next().unwrap();
        assert!(r2.is_some());
        assert_eq!(r2.unwrap()[0], SochValue::Int(2));

        let r3 = node.next().unwrap();
        assert!(r3.is_none());
    }

    #[test]
    fn test_values_node_reset() {
        let rows = vec![vec![SochValue::Int(42)]];
        let schema = Schema::new(vec![ColumnMeta::new("x")]);
        let mut node = ValuesNode::new(schema, rows);

        assert!(node.next().unwrap().is_some());
        assert!(node.next().unwrap().is_none());

        node.reset().unwrap();
        assert!(node.next().unwrap().is_some());
        assert!(node.next().unwrap().is_none());
    }

    #[test]
    fn test_values_node_collect_all() {
        let rows = sample_rows();
        let mut node = ValuesNode::new(schema_id_name_age(), rows.clone());
        let collected = node.collect_all().unwrap();
        assert_eq!(collected.len(), 5);
        assert_eq!(collected, rows);
    }

    // ========================================================================
    // EmptyNode tests
    // ========================================================================

    #[test]
    fn test_empty_node() {
        let mut node = EmptyNode::new(schema_abc());
        assert!(node.next().unwrap().is_none());
        assert_eq!(node.schema().len(), 3);
    }

    // ========================================================================
    // FilterNode tests
    // ========================================================================

    #[test]
    fn test_filter_equality() {
        // Filter: age = 25
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::Column(ColumnRef::new("age"))),
            op: BinaryOperator::Eq,
            right: Box::new(Expr::Literal(Literal::Integer(25))),
        };

        let input = values_node(schema_id_name_age(), sample_rows());
        let mut filter = FilterNode::new(input, predicate);

        let rows = filter.collect_all().unwrap();
        assert_eq!(rows.len(), 2); // Bob and Dave are both 25
        assert_eq!(rows[0][1], SochValue::Text("Bob".into()));
        assert_eq!(rows[1][1], SochValue::Text("Dave".into()));
    }

    #[test]
    fn test_filter_greater_than() {
        // Filter: age > 28
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::Column(ColumnRef::new("age"))),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(Literal::Integer(28))),
        };

        let input = values_node(schema_id_name_age(), sample_rows());
        let mut filter = FilterNode::new(input, predicate);

        let rows = filter.collect_all().unwrap();
        assert_eq!(rows.len(), 3); // Alice(30), Carol(35), Eve(30)
    }

    #[test]
    fn test_filter_or_condition() {
        // Filter: age = 25 OR age = 35
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("age"))),
                op: BinaryOperator::Eq,
                right: Box::new(Expr::Literal(Literal::Integer(25))),
            }),
            op: BinaryOperator::Or,
            right: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("age"))),
                op: BinaryOperator::Eq,
                right: Box::new(Expr::Literal(Literal::Integer(35))),
            }),
        };

        let input = values_node(schema_id_name_age(), sample_rows());
        let mut filter = FilterNode::new(input, predicate);

        let rows = filter.collect_all().unwrap();
        assert_eq!(rows.len(), 3); // Bob(25), Carol(35), Dave(25)
    }

    #[test]
    fn test_filter_and_condition() {
        // Filter: age >= 30 AND name != 'Eve'
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("age"))),
                op: BinaryOperator::Ge,
                right: Box::new(Expr::Literal(Literal::Integer(30))),
            }),
            op: BinaryOperator::And,
            right: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("name"))),
                op: BinaryOperator::Ne,
                right: Box::new(Expr::Literal(Literal::String("Eve".into()))),
            }),
        };

        let input = values_node(schema_id_name_age(), sample_rows());
        let mut filter = FilterNode::new(input, predicate);

        let rows = filter.collect_all().unwrap();
        assert_eq!(rows.len(), 2); // Alice(30), Carol(35)
    }

    #[test]
    fn test_filter_no_matches() {
        // Filter: age > 100
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::Column(ColumnRef::new("age"))),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(Literal::Integer(100))),
        };

        let input = values_node(schema_id_name_age(), sample_rows());
        let mut filter = FilterNode::new(input, predicate);

        let rows = filter.collect_all().unwrap();
        assert_eq!(rows.len(), 0);
    }

    // ========================================================================
    // ProjectNode tests
    // ========================================================================

    #[test]
    fn test_project_select_columns() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let mut proj = ProjectNode::columns(input, vec!["name".into(), "age".into()]);

        let rows = proj.collect_all().unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(proj.schema().len(), 2);
        assert_eq!(proj.schema().column_names(), vec!["name", "age"]);
        assert_eq!(rows[0], vec![SochValue::Text("Alice".into()), SochValue::Int(30)]);
    }

    #[test]
    fn test_project_expression() {
        // Project: age + 10 AS age_plus_10
        let input = values_node(schema_id_name_age(), sample_rows());
        let exprs = vec![ProjectExpr {
            expr: Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("age"))),
                op: BinaryOperator::Plus,
                right: Box::new(Expr::Literal(Literal::Integer(10))),
            },
            alias: "age_plus_10".into(),
        }];
        let mut proj = ProjectNode::new(input, exprs);

        let rows = proj.collect_all().unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0][0], SochValue::Int(40)); // Alice: 30 + 10
        assert_eq!(rows[1][0], SochValue::Int(35)); // Bob: 25 + 10
    }

    #[test]
    fn test_project_multiply() {
        // Project: age * 2 AS double_age
        let input = values_node(schema_id_name_age(), sample_rows());
        let exprs = vec![ProjectExpr {
            expr: Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("age"))),
                op: BinaryOperator::Multiply,
                right: Box::new(Expr::Literal(Literal::Integer(2))),
            },
            alias: "double_age".into(),
        }];
        let mut proj = ProjectNode::new(input, exprs);

        let rows = proj.collect_all().unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0][0], SochValue::Int(60)); // Alice: 30 * 2
        assert_eq!(rows[1][0], SochValue::Int(50)); // Bob: 25 * 2
    }

    #[test]
    fn test_project_literal() {
        // Project: 'constant' AS tag
        let input = values_node(schema_id_name_age(), sample_rows());
        let exprs = vec![ProjectExpr {
            expr: Expr::Literal(Literal::String("hello".into())),
            alias: "tag".into(),
        }];
        let mut proj = ProjectNode::new(input, exprs);

        let rows = proj.collect_all().unwrap();
        assert_eq!(rows.len(), 5);
        for row in &rows {
            assert_eq!(row[0], SochValue::Text("hello".into()));
        }
    }

    // ========================================================================
    // PassThroughNode tests
    // ========================================================================

    #[test]
    fn test_passthrough() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let mut pass = PassThroughNode::new(input);
        let rows = pass.collect_all().unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(pass.schema().len(), 3);
    }

    // ========================================================================
    // SortNode tests
    // ========================================================================

    #[test]
    fn test_sort_ascending() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let sort_keys = vec![SortKey {
            expr: Expr::Column(ColumnRef::new("name")),
            ascending: true,
            nulls_first: true,
        }];
        let mut sort = SortNode::new(input, sort_keys);

        let rows = sort.collect_all().unwrap();
        assert_eq!(rows.len(), 5);
        // Sorted alphabetically: Alice, Bob, Carol, Dave, Eve
        assert_eq!(rows[0][1], SochValue::Text("Alice".into()));
        assert_eq!(rows[1][1], SochValue::Text("Bob".into()));
        assert_eq!(rows[2][1], SochValue::Text("Carol".into()));
        assert_eq!(rows[3][1], SochValue::Text("Dave".into()));
        assert_eq!(rows[4][1], SochValue::Text("Eve".into()));
    }

    #[test]
    fn test_sort_descending() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let sort_keys = vec![SortKey {
            expr: Expr::Column(ColumnRef::new("age")),
            ascending: false,
            nulls_first: false,
        }];
        let mut sort = SortNode::new(input, sort_keys);

        let rows = sort.collect_all().unwrap();
        assert_eq!(rows.len(), 5);
        // Ages: 35, 30, 30, 25, 25 (descending)
        assert_eq!(rows[0][2], SochValue::Int(35)); // Carol
        // 30s are Alice or Eve
        assert!(rows[1][2] == SochValue::Int(30));
        assert!(rows[2][2] == SochValue::Int(30));
        assert_eq!(rows[3][2], SochValue::Int(25));
        assert_eq!(rows[4][2], SochValue::Int(25));
    }

    #[test]
    fn test_sort_with_nulls() {
        let schema = Schema::new(vec![ColumnMeta::new("x")]);
        let rows = vec![
            vec![SochValue::Int(3)],
            vec![SochValue::Null],
            vec![SochValue::Int(1)],
            vec![SochValue::Null],
            vec![SochValue::Int(2)],
        ];

        // Nulls first, ascending
        let input = values_node(schema.clone(), rows.clone());
        let sort_keys = vec![SortKey {
            expr: Expr::Column(ColumnRef::new("x")),
            ascending: true,
            nulls_first: true,
        }];
        let mut sort = SortNode::new(input, sort_keys);
        let result = sort.collect_all().unwrap();
        assert_eq!(result[0][0], SochValue::Null);
        assert_eq!(result[1][0], SochValue::Null);
        assert_eq!(result[2][0], SochValue::Int(1));
        assert_eq!(result[3][0], SochValue::Int(2));
        assert_eq!(result[4][0], SochValue::Int(3));
    }

    // ========================================================================
    // LimitNode tests
    // ========================================================================

    #[test]
    fn test_limit_only() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let mut limit = LimitNode::new(input, Some(3), 0);

        let rows = limit.collect_all().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0], SochValue::Int(1));
        assert_eq!(rows[2][0], SochValue::Int(3));
    }

    #[test]
    fn test_offset_only() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let mut limit = LimitNode::new(input, None, 3);

        let rows = limit.collect_all().unwrap();
        assert_eq!(rows.len(), 2); // Skip 3, get 4 and 5
        assert_eq!(rows[0][0], SochValue::Int(4));
        assert_eq!(rows[1][0], SochValue::Int(5));
    }

    #[test]
    fn test_limit_and_offset() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let mut limit = LimitNode::new(input, Some(2), 1);

        let rows = limit.collect_all().unwrap();
        assert_eq!(rows.len(), 2); // Skip 1, take 2
        assert_eq!(rows[0][0], SochValue::Int(2)); // Bob
        assert_eq!(rows[1][0], SochValue::Int(3)); // Carol
    }

    #[test]
    fn test_limit_larger_than_input() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let mut limit = LimitNode::new(input, Some(100), 0);

        let rows = limit.collect_all().unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_offset_past_end() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let mut limit = LimitNode::new(input, None, 100);

        let rows = limit.collect_all().unwrap();
        assert_eq!(rows.len(), 0);
    }

    // ========================================================================
    // HashAggregateNode tests
    // ========================================================================

    #[test]
    fn test_aggregate_count_all() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let aggs = vec![AggDef {
            func: AggFunc::Count,
            expr: None, // COUNT(*)
            alias: "count".into(),
        }];
        let mut agg = HashAggregateNode::new(input, vec![], aggs);

        let rows = agg.collect_all().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], SochValue::Int(5));
    }

    #[test]
    fn test_aggregate_sum_avg() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let aggs = vec![
            AggDef {
                func: AggFunc::Sum,
                expr: Some(Expr::Column(ColumnRef::new("age"))),
                alias: "total_age".into(),
            },
            AggDef {
                func: AggFunc::Avg,
                expr: Some(Expr::Column(ColumnRef::new("age"))),
                alias: "avg_age".into(),
            },
        ];
        let mut agg = HashAggregateNode::new(input, vec![], aggs);

        let rows = agg.collect_all().unwrap();
        assert_eq!(rows.len(), 1);
        // Sum: 30 + 25 + 35 + 25 + 30 = 145
        assert_eq!(rows[0][0], SochValue::Int(145));
        // Avg: 145 / 5 = 29.0
        assert_eq!(rows[0][1], SochValue::Float(29.0));
    }

    #[test]
    fn test_aggregate_min_max() {
        let input = values_node(schema_id_name_age(), sample_rows());
        let aggs = vec![
            AggDef {
                func: AggFunc::Min,
                expr: Some(Expr::Column(ColumnRef::new("age"))),
                alias: "min_age".into(),
            },
            AggDef {
                func: AggFunc::Max,
                expr: Some(Expr::Column(ColumnRef::new("age"))),
                alias: "max_age".into(),
            },
        ];
        let mut agg = HashAggregateNode::new(input, vec![], aggs);

        let rows = agg.collect_all().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], SochValue::Int(25));
        assert_eq!(rows[0][1], SochValue::Int(35));
    }

    #[test]
    fn test_aggregate_group_by() {
        // GROUP BY age, COUNT(*), SUM(id)
        let input = values_node(schema_id_name_age(), sample_rows());
        let group_by = vec![Expr::Column(ColumnRef::new("age"))];
        let aggs = vec![
            AggDef {
                func: AggFunc::Count,
                expr: None,
                alias: "cnt".into(),
            },
            AggDef {
                func: AggFunc::Sum,
                expr: Some(Expr::Column(ColumnRef::new("id"))),
                alias: "sum_id".into(),
            },
        ];
        let mut agg = HashAggregateNode::new(input, group_by, aggs);

        let mut rows = agg.collect_all().unwrap();
        // 3 distinct ages: 25, 30, 35
        assert_eq!(rows.len(), 3);

        // Sort by first column (group key = age) for deterministic comparison
        rows.sort_by(|a, b| {
            if let (SochValue::Int(ia), SochValue::Int(ib)) = (&a[0], &b[0]) {
                ia.cmp(ib)
            } else {
                std::cmp::Ordering::Equal
            }
        });

        // age=25: Bob(2) + Dave(4) → count=2, sum_id=6
        assert_eq!(rows[0][0], SochValue::Int(25));
        assert_eq!(rows[0][1], SochValue::Int(2));
        assert_eq!(rows[0][2], SochValue::Int(6));

        // age=30: Alice(1) + Eve(5) → count=2, sum_id=6
        assert_eq!(rows[1][0], SochValue::Int(30));
        assert_eq!(rows[1][1], SochValue::Int(2));
        assert_eq!(rows[1][2], SochValue::Int(6));

        // age=35: Carol(3) → count=1, sum_id=3
        assert_eq!(rows[2][0], SochValue::Int(35));
        assert_eq!(rows[2][1], SochValue::Int(1));
        assert_eq!(rows[2][2], SochValue::Int(3));
    }

    #[test]
    fn test_aggregate_empty_input() {
        // COUNT(*) on empty input should return 1 row with count=0
        let input = values_node(schema_id_name_age(), vec![]);
        let aggs = vec![AggDef {
            func: AggFunc::Count,
            expr: None,
            alias: "count".into(),
        }];
        let mut agg = HashAggregateNode::new(input, vec![], aggs);

        let rows = agg.collect_all().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], SochValue::Int(0));
    }

    // ========================================================================
    // Composite pipeline tests
    // ========================================================================

    #[test]
    fn test_filter_then_sort_then_limit() {
        // SELECT * FROM data WHERE age >= 30 ORDER BY name ASC LIMIT 2
        let input = values_node(schema_id_name_age(), sample_rows());

        // Filter: age >= 30
        let filter = FilterNode::new(
            input,
            Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("age"))),
                op: BinaryOperator::Ge,
                right: Box::new(Expr::Literal(Literal::Integer(30))),
            },
        );

        // Sort: name ASC
        let sort = SortNode::new(
            Box::new(filter),
            vec![SortKey {
                expr: Expr::Column(ColumnRef::new("name")),
                ascending: true,
                nulls_first: true,
            }],
        );

        // Limit: 2
        let mut limit = LimitNode::new(Box::new(sort), Some(2), 0);

        let rows = limit.collect_all().unwrap();
        assert_eq!(rows.len(), 2);
        // age >= 30: Alice(30), Carol(35), Eve(30) → sorted: Alice, Carol, Eve → limit 2: Alice, Carol
        assert_eq!(rows[0][1], SochValue::Text("Alice".into()));
        assert_eq!(rows[1][1], SochValue::Text("Carol".into()));
    }

    #[test]
    fn test_project_with_filter() {
        // SELECT name, age + 1 AS next_age FROM data WHERE id < 3
        let input = values_node(schema_id_name_age(), sample_rows());

        // Filter: id < 3
        let filter = FilterNode::new(
            input,
            Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("id"))),
                op: BinaryOperator::Lt,
                right: Box::new(Expr::Literal(Literal::Integer(3))),
            },
        );

        // Project: name, age + 1 AS next_age
        let exprs = vec![
            ProjectExpr {
                expr: Expr::Column(ColumnRef::new("name")),
                alias: "name".into(),
            },
            ProjectExpr {
                expr: Expr::BinaryOp {
                    left: Box::new(Expr::Column(ColumnRef::new("age"))),
                    op: BinaryOperator::Plus,
                    right: Box::new(Expr::Literal(Literal::Integer(1))),
                },
                alias: "next_age".into(),
            },
        ];
        let mut proj = ProjectNode::new(Box::new(filter), exprs);

        let rows = proj.collect_all().unwrap();
        assert_eq!(rows.len(), 2); // id=1 (Alice) and id=2 (Bob)
        assert_eq!(rows[0], vec![SochValue::Text("Alice".into()), SochValue::Int(31)]);
        assert_eq!(rows[1], vec![SochValue::Text("Bob".into()), SochValue::Int(26)]);
    }

    #[test]
    fn test_aggregate_with_filter() {
        // SELECT age, COUNT(*) FROM data WHERE age < 35 GROUP BY age
        let input = values_node(schema_id_name_age(), sample_rows());

        // Filter: age < 35
        let filter = FilterNode::new(
            input,
            Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("age"))),
                op: BinaryOperator::Lt,
                right: Box::new(Expr::Literal(Literal::Integer(35))),
            },
        );

        let group_by = vec![Expr::Column(ColumnRef::new("age"))];
        let aggs = vec![AggDef {
            func: AggFunc::Count,
            expr: None,
            alias: "cnt".into(),
        }];
        let mut agg = HashAggregateNode::new(Box::new(filter), group_by, aggs);

        let mut rows = agg.collect_all().unwrap();
        assert_eq!(rows.len(), 2); // age=25 and age=30 (Carol excluded)

        rows.sort_by(|a, b| {
            if let (SochValue::Int(ia), SochValue::Int(ib)) = (&a[0], &b[0]) {
                ia.cmp(ib)
            } else {
                std::cmp::Ordering::Equal
            }
        });

        assert_eq!(rows[0], vec![SochValue::Int(25), SochValue::Int(2)]);
        assert_eq!(rows[1], vec![SochValue::Int(30), SochValue::Int(2)]);
    }

    // ========================================================================
    // HashJoinNode tests
    // ========================================================================

    #[test]
    fn test_hash_join_inner() {
        // Users: (id, name)
        let users_schema = Schema::new(vec![
            ColumnMeta::qualified("users", "id"),
            ColumnMeta::qualified("users", "name"),
        ]);
        let users = vec![
            vec![SochValue::Int(1), SochValue::Text("Alice".into())],
            vec![SochValue::Int(2), SochValue::Text("Bob".into())],
            vec![SochValue::Int(3), SochValue::Text("Carol".into())],
        ];

        // Orders: (user_id, product)
        let orders_schema = Schema::new(vec![
            ColumnMeta::qualified("orders", "user_id"),
            ColumnMeta::qualified("orders", "product"),
        ]);
        let orders = vec![
            vec![SochValue::Int(1), SochValue::Text("Widget".into())],
            vec![SochValue::Int(2), SochValue::Text("Gadget".into())],
            vec![SochValue::Int(1), SochValue::Text("Doohickey".into())],
        ];

        let left = values_node(users_schema, users);
        let right = values_node(orders_schema, orders);

        // Join ON users.id = orders.user_id
        // HashJoin needs separate build key and probe key expressions
        let build_key = Expr::Column(ColumnRef::qualified("users", "id"));
        let probe_key = Expr::Column(ColumnRef::qualified("orders", "user_id"));

        let mut join = HashJoinNode::new(
            left,
            right,
            build_key,
            probe_key,
            JoinType::Inner,
        );

        let mut rows = join.collect_all().unwrap();
        assert_eq!(rows.len(), 3); // Alice×Widget, Alice×Doohickey, Bob×Gadget

        // Sort for determinism
        rows.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

        // Schema should be users.id, users.name, orders.user_id, orders.product
        assert_eq!(join.schema().len(), 4);
    }

    // ========================================================================
    // Eval tests (expression evaluator)
    // ========================================================================

    #[test]
    fn test_eval_literal() {
        use crate::executor::eval::eval_expr;
        let schema = Schema::new(vec![]);
        let row = vec![];

        assert_eq!(
            eval_expr(&Expr::Literal(Literal::Integer(42)), &row, &schema).unwrap(),
            SochValue::Int(42)
        );
        assert_eq!(
            eval_expr(&Expr::Literal(Literal::Float(3.14)), &row, &schema).unwrap(),
            SochValue::Float(3.14)
        );
        assert_eq!(
            eval_expr(&Expr::Literal(Literal::String("hi".into())), &row, &schema).unwrap(),
            SochValue::Text("hi".into())
        );
        assert_eq!(
            eval_expr(&Expr::Literal(Literal::Boolean(true)), &row, &schema).unwrap(),
            SochValue::Bool(true)
        );
        assert_eq!(
            eval_expr(&Expr::Literal(Literal::Null), &row, &schema).unwrap(),
            SochValue::Null
        );
    }

    #[test]
    fn test_eval_column_ref() {
        use crate::executor::eval::eval_expr;
        let schema = Schema::new(vec![
            ColumnMeta::new("x"),
            ColumnMeta::new("y"),
        ]);
        let row = vec![SochValue::Int(10), SochValue::Text("hello".into())];

        assert_eq!(
            eval_expr(&Expr::Column(ColumnRef::new("x")), &row, &schema).unwrap(),
            SochValue::Int(10)
        );
        assert_eq!(
            eval_expr(&Expr::Column(ColumnRef::new("y")), &row, &schema).unwrap(),
            SochValue::Text("hello".into())
        );
    }

    #[test]
    fn test_eval_arithmetic() {
        use crate::executor::eval::eval_expr;
        let schema = Schema::new(vec![ColumnMeta::new("a")]);
        let row = vec![SochValue::Int(10)];

        // a + 5
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Column(ColumnRef::new("a"))),
            op: BinaryOperator::Plus,
            right: Box::new(Expr::Literal(Literal::Integer(5))),
        };
        assert_eq!(eval_expr(&expr, &row, &schema).unwrap(), SochValue::Int(15));

        // a * 3
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Column(ColumnRef::new("a"))),
            op: BinaryOperator::Multiply,
            right: Box::new(Expr::Literal(Literal::Integer(3))),
        };
        assert_eq!(eval_expr(&expr, &row, &schema).unwrap(), SochValue::Int(30));
    }

    #[test]
    fn test_eval_comparison() {
        use crate::executor::eval::eval_predicate;
        let schema = Schema::new(vec![ColumnMeta::new("x")]);
        let row = vec![SochValue::Int(10)];

        // x = 10
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Column(ColumnRef::new("x"))),
            op: BinaryOperator::Eq,
            right: Box::new(Expr::Literal(Literal::Integer(10))),
        };
        assert!(eval_predicate(&expr, &row, &schema).unwrap());

        // x < 5
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Column(ColumnRef::new("x"))),
            op: BinaryOperator::Lt,
            right: Box::new(Expr::Literal(Literal::Integer(5))),
        };
        assert!(!eval_predicate(&expr, &row, &schema).unwrap());
    }

    #[test]
    fn test_eval_is_null() {
        use crate::executor::eval::eval_predicate;
        let schema = Schema::new(vec![ColumnMeta::new("x")]);

        let null_row = vec![SochValue::Null];
        let non_null_row = vec![SochValue::Int(5)];

        let is_null = Expr::IsNull {
            expr: Box::new(Expr::Column(ColumnRef::new("x"))),
            negated: false,
        };
        assert!(eval_predicate(&is_null, &null_row, &schema).unwrap());
        assert!(!eval_predicate(&is_null, &non_null_row, &schema).unwrap());

        let is_not_null = Expr::IsNull {
            expr: Box::new(Expr::Column(ColumnRef::new("x"))),
            negated: true,
        };
        assert!(!eval_predicate(&is_not_null, &null_row, &schema).unwrap());
        assert!(eval_predicate(&is_not_null, &non_null_row, &schema).unwrap());
    }

    #[test]
    fn test_eval_between() {
        use crate::executor::eval::eval_predicate;
        let schema = Schema::new(vec![ColumnMeta::new("x")]);
        let row = vec![SochValue::Int(5)];

        // x BETWEEN 1 AND 10
        let between = Expr::Between {
            expr: Box::new(Expr::Column(ColumnRef::new("x"))),
            low: Box::new(Expr::Literal(Literal::Integer(1))),
            high: Box::new(Expr::Literal(Literal::Integer(10))),
            negated: false,
        };
        assert!(eval_predicate(&between, &row, &schema).unwrap());

        // x NOT BETWEEN 1 AND 3
        let not_between = Expr::Between {
            expr: Box::new(Expr::Column(ColumnRef::new("x"))),
            low: Box::new(Expr::Literal(Literal::Integer(1))),
            high: Box::new(Expr::Literal(Literal::Integer(3))),
            negated: true,
        };
        assert!(eval_predicate(&not_between, &row, &schema).unwrap());
    }

    #[test]
    fn test_eval_in_list() {
        use crate::executor::eval::eval_predicate;
        let schema = Schema::new(vec![ColumnMeta::new("x")]);
        let row = vec![SochValue::Int(3)];

        // x IN (1, 2, 3)
        let in_list = Expr::InList {
            expr: Box::new(Expr::Column(ColumnRef::new("x"))),
            list: vec![
                Expr::Literal(Literal::Integer(1)),
                Expr::Literal(Literal::Integer(2)),
                Expr::Literal(Literal::Integer(3)),
            ],
            negated: false,
        };
        assert!(eval_predicate(&in_list, &row, &schema).unwrap());

        // x IN (10, 20)
        let not_in = Expr::InList {
            expr: Box::new(Expr::Column(ColumnRef::new("x"))),
            list: vec![
                Expr::Literal(Literal::Integer(10)),
                Expr::Literal(Literal::Integer(20)),
            ],
            negated: false,
        };
        assert!(!eval_predicate(&not_in, &row, &schema).unwrap());
    }

    #[test]
    fn test_eval_like() {
        use crate::executor::eval::eval_predicate;
        let schema = Schema::new(vec![ColumnMeta::new("name")]);
        let row = vec![SochValue::Text("Alice".into())];

        // name LIKE 'Ali%'
        let like = Expr::Like {
            expr: Box::new(Expr::Column(ColumnRef::new("name"))),
            pattern: Box::new(Expr::Literal(Literal::String("Ali%".into()))),
            escape: None,
            negated: false,
        };
        assert!(eval_predicate(&like, &row, &schema).unwrap());

        // name LIKE '%ob'
        let not_match = Expr::Like {
            expr: Box::new(Expr::Column(ColumnRef::new("name"))),
            pattern: Box::new(Expr::Literal(Literal::String("%ob".into()))),
            escape: None,
            negated: false,
        };
        assert!(!eval_predicate(&not_match, &row, &schema).unwrap());
    }

    // ========================================================================
    // Integration test: full pipeline
    // ========================================================================

    #[test]
    fn test_full_pipeline_filter_project_sort_limit() {
        // Simulates: SELECT name, age FROM users WHERE age > 25 ORDER BY age DESC LIMIT 2
        let input = values_node(schema_id_name_age(), sample_rows());

        // 1. Filter: age > 25
        let filter = FilterNode::new(
            input,
            Expr::BinaryOp {
                left: Box::new(Expr::Column(ColumnRef::new("age"))),
                op: BinaryOperator::Gt,
                right: Box::new(Expr::Literal(Literal::Integer(25))),
            },
        );

        // 2. Project: name, age
        let project = ProjectNode::columns(
            Box::new(filter),
            vec!["name".into(), "age".into()],
        );

        // 3. Sort: age DESC
        let sort = SortNode::new(
            Box::new(project),
            vec![SortKey {
                expr: Expr::Column(ColumnRef::new("age")),
                ascending: false,
                nulls_first: false,
            }],
        );

        // 4. Limit: 2
        let mut limit = LimitNode::new(Box::new(sort), Some(2), 0);

        let rows = limit.collect_all().unwrap();
        assert_eq!(rows.len(), 2);
        // age > 25: Alice(30), Carol(35), Eve(30) → sorted desc: Carol(35), Alice/Eve(30) → limit 2
        assert_eq!(rows[0][1], SochValue::Int(35)); // Carol
        assert_eq!(rows[1][1], SochValue::Int(30)); // Alice or Eve
    }
}

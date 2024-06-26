// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array};
use arrow::compute::SortOptions;
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use arrow_schema::Schema;

use datafusion_common::ScalarValue;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::PhysicalExprRef;

use rand::Rng;

use datafusion::common::JoinSide;
use datafusion::logical_expr::{JoinType, Operator};
use datafusion::physical_expr::expressions::BinaryExpr;
use datafusion::physical_plan::collect;
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::joins::{
    HashJoinExec, NestedLoopJoinExec, PartitionMode, SortMergeJoinExec,
};
use datafusion::physical_plan::memory::MemoryExec;

use datafusion::prelude::{SessionConfig, SessionContext};
use test_utils::stagger_batch_with_seed;

#[tokio::test]
async fn test_inner_join_1k() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Inner,
        None,
    )
    .run_test()
    .await
}

fn less_than_10_join_filter(schema1: Arc<Schema>, _schema2: Arc<Schema>) -> JoinFilter {
    let less_than_100 = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("a", 0)),
        Operator::Lt,
        Arc::new(Literal::new(ScalarValue::from(100))),
    )) as _;
    let column_indices = vec![ColumnIndex {
        index: 0,
        side: JoinSide::Left,
    }];
    let intermediate_schema =
        Schema::new(vec![schema1.field_with_name("a").unwrap().to_owned()]);

    JoinFilter::new(less_than_100, column_indices, intermediate_schema)
}

#[tokio::test]
async fn test_inner_join_1k_filtered() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Inner,
        Some(Box::new(less_than_10_join_filter)),
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_inner_join_1k_smjoin() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Inner,
        None,
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_left_join_1k() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Left,
        None,
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_left_join_1k_filtered() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Left,
        Some(Box::new(less_than_10_join_filter)),
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_right_join_1k() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Right,
        None,
    )
    .run_test()
    .await
}
// Add support for Right filtered joins
#[ignore]
#[tokio::test]
async fn test_right_join_1k_filtered() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Right,
        Some(Box::new(less_than_10_join_filter)),
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_full_join_1k() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Full,
        None,
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_full_join_1k_filtered() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::Full,
        Some(Box::new(less_than_10_join_filter)),
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_semi_join_1k() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::LeftSemi,
        None,
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_semi_join_1k_filtered() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::LeftSemi,
        Some(Box::new(less_than_10_join_filter)),
    )
    .run_test()
    .await
}

#[tokio::test]
async fn test_anti_join_1k() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::LeftAnti,
        None,
    )
    .run_test()
    .await
}

// Test failed for now. https://github.com/apache/datafusion/issues/10872
#[ignore]
#[tokio::test]
async fn test_anti_join_1k_filtered() {
    JoinFuzzTestCase::new(
        make_staggered_batches(1000),
        make_staggered_batches(1000),
        JoinType::LeftAnti,
        Some(Box::new(less_than_10_join_filter)),
    )
    .run_test()
    .await
}

type JoinFilterBuilder = Box<dyn Fn(Arc<Schema>, Arc<Schema>) -> JoinFilter>;

struct JoinFuzzTestCase {
    batch_sizes: &'static [usize],
    input1: Vec<RecordBatch>,
    input2: Vec<RecordBatch>,
    join_type: JoinType,
    join_filter_builder: Option<JoinFilterBuilder>,
}

impl JoinFuzzTestCase {
    fn new(
        input1: Vec<RecordBatch>,
        input2: Vec<RecordBatch>,
        join_type: JoinType,
        join_filter_builder: Option<JoinFilterBuilder>,
    ) -> Self {
        Self {
            batch_sizes: &[1, 2, 7, 49, 50, 51, 100],
            input1,
            input2,
            join_type,
            join_filter_builder,
        }
    }

    fn column_indices(&self) -> Vec<ColumnIndex> {
        vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ]
    }

    fn on_columns(&self) -> Vec<(PhysicalExprRef, PhysicalExprRef)> {
        let schema1 = self.input1[0].schema();
        let schema2 = self.input2[0].schema();
        vec![
            (
                Arc::new(Column::new_with_schema("a", &schema1).unwrap()) as _,
                Arc::new(Column::new_with_schema("a", &schema2).unwrap()) as _,
            ),
            (
                Arc::new(Column::new_with_schema("b", &schema1).unwrap()) as _,
                Arc::new(Column::new_with_schema("b", &schema2).unwrap()) as _,
            ),
        ]
    }

    fn intermediate_schema(&self) -> Schema {
        let schema1 = self.input1[0].schema();
        let schema2 = self.input2[0].schema();
        Schema::new(vec![
            schema1
                .field_with_name("a")
                .unwrap()
                .to_owned()
                .with_nullable(true),
            schema1
                .field_with_name("b")
                .unwrap()
                .to_owned()
                .with_nullable(true),
            schema2.field_with_name("a").unwrap().to_owned(),
            schema2.field_with_name("b").unwrap().to_owned(),
        ])
    }

    fn left_right(&self) -> (Arc<MemoryExec>, Arc<MemoryExec>) {
        let schema1 = self.input1[0].schema();
        let schema2 = self.input2[0].schema();
        let left = Arc::new(
            MemoryExec::try_new(&[self.input1.clone()], schema1.clone(), None).unwrap(),
        );
        let right = Arc::new(
            MemoryExec::try_new(&[self.input2.clone()], schema2.clone(), None).unwrap(),
        );
        (left, right)
    }

    fn join_filter(&self) -> Option<JoinFilter> {
        let schema1 = self.input1[0].schema();
        let schema2 = self.input2[0].schema();
        self.join_filter_builder
            .as_ref()
            .map(|builder| builder(schema1, schema2))
    }

    fn sort_merge_join(&self) -> Arc<SortMergeJoinExec> {
        let (left, right) = self.left_right();
        Arc::new(
            SortMergeJoinExec::try_new(
                left,
                right,
                self.on_columns().clone(),
                self.join_filter(),
                self.join_type,
                vec![SortOptions::default(), SortOptions::default()],
                false,
            )
            .unwrap(),
        )
    }

    fn hash_join(&self) -> Arc<HashJoinExec> {
        let (left, right) = self.left_right();
        Arc::new(
            HashJoinExec::try_new(
                left,
                right,
                self.on_columns().clone(),
                self.join_filter(),
                &self.join_type,
                None,
                PartitionMode::Partitioned,
                false,
            )
            .unwrap(),
        )
    }

    fn nested_loop_join(&self) -> Arc<NestedLoopJoinExec> {
        let (left, right) = self.left_right();
        // Nested loop join uses filter for joining records
        let column_indices = self.column_indices();
        let intermediate_schema = self.intermediate_schema();

        let equal_a = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Column::new("a", 2)),
        )) as _;
        let equal_b = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Eq,
            Arc::new(Column::new("b", 3)),
        )) as _;
        let expression = Arc::new(BinaryExpr::new(equal_a, Operator::And, equal_b)) as _;

        let on_filter = JoinFilter::new(expression, column_indices, intermediate_schema);

        Arc::new(
            NestedLoopJoinExec::try_new(left, right, Some(on_filter), &self.join_type)
                .unwrap(),
        )
    }

    /// Perform sort-merge join and hash join on same input
    /// and verify two outputs are equal
    async fn run_test(&self) {
        for batch_size in self.batch_sizes {
            let session_config = SessionConfig::new().with_batch_size(*batch_size);
            let ctx = SessionContext::new_with_config(session_config);
            let task_ctx = ctx.task_ctx();
            let smj = self.sort_merge_join();
            let smj_collected = collect(smj, task_ctx.clone()).await.unwrap();

            let hj = self.hash_join();
            let hj_collected = collect(hj, task_ctx.clone()).await.unwrap();

            let nlj = self.nested_loop_join();
            let nlj_collected = collect(nlj, task_ctx.clone()).await.unwrap();

            // compare
            let smj_formatted =
                pretty_format_batches(&smj_collected).unwrap().to_string();
            let hj_formatted = pretty_format_batches(&hj_collected).unwrap().to_string();
            let nlj_formatted =
                pretty_format_batches(&nlj_collected).unwrap().to_string();

            let mut smj_formatted_sorted: Vec<&str> =
                smj_formatted.trim().lines().collect();
            smj_formatted_sorted.sort_unstable();

            let mut hj_formatted_sorted: Vec<&str> =
                hj_formatted.trim().lines().collect();
            hj_formatted_sorted.sort_unstable();

            let mut nlj_formatted_sorted: Vec<&str> =
                nlj_formatted.trim().lines().collect();
            nlj_formatted_sorted.sort_unstable();

            assert_eq!(
                smj_formatted_sorted.len(),
                hj_formatted_sorted.len(),
                "SortMergeJoinExec and HashJoinExec produced different row counts"
            );
            for (i, (smj_line, hj_line)) in smj_formatted_sorted
                .iter()
                .zip(&hj_formatted_sorted)
                .enumerate()
            {
                assert_eq!(
                    (i, smj_line),
                    (i, hj_line),
                    "SortMergeJoinExec and HashJoinExec produced different results"
                );
            }

            for (i, (nlj_line, hj_line)) in nlj_formatted_sorted
                .iter()
                .zip(&hj_formatted_sorted)
                .enumerate()
            {
                assert_eq!(
                    (i, nlj_line),
                    (i, hj_line),
                    "NestedLoopJoinExec and HashJoinExec produced different results"
                );
            }
        }
    }
}

/// Return randomly sized record batches with:
/// two sorted int32 columns 'a', 'b' ranged from 0..99 as join columns
/// two random int32 columns 'x', 'y' as other columns
fn make_staggered_batches(len: usize) -> Vec<RecordBatch> {
    let mut rng = rand::thread_rng();
    let mut input12: Vec<(i32, i32)> = vec![(0, 0); len];
    let mut input3: Vec<i32> = vec![0; len];
    let mut input4: Vec<i32> = vec![0; len];
    input12
        .iter_mut()
        .for_each(|v| *v = (rng.gen_range(0..100), rng.gen_range(0..100)));
    rng.fill(&mut input3[..]);
    rng.fill(&mut input4[..]);
    input12.sort_unstable();
    let input1 = Int32Array::from_iter_values(input12.clone().into_iter().map(|k| k.0));
    let input2 = Int32Array::from_iter_values(input12.clone().into_iter().map(|k| k.1));
    let input3 = Int32Array::from_iter_values(input3);
    let input4 = Int32Array::from_iter_values(input4);

    // split into several record batches
    let batch = RecordBatch::try_from_iter(vec![
        ("a", Arc::new(input1) as ArrayRef),
        ("b", Arc::new(input2) as ArrayRef),
        ("x", Arc::new(input3) as ArrayRef),
        ("y", Arc::new(input4) as ArrayRef),
    ])
    .unwrap();

    // use a random number generator to pick a random sized output
    stagger_batch_with_seed(batch, 42)
}

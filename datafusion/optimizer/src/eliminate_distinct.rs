
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

//! [`EliminateDistinctFromMinMax`] Removes redundant distinct in min and max

use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::{internal_err, Result};
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::logical_plan::LogicalPlan;
use datafusion_expr::{Aggregate, Expr};
use std::sync::OnceLock;

/// Optimization rule that eliminate redundant distinct in min and max expr.
#[derive(Default)]
pub struct EliminateDistinct;

impl EliminateDistinct {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}
static WORKSPACE_ROOT_LOCK: OnceLock<Vec<String>> = OnceLock::new();

fn rewrite_aggr_expr(expr:Expr) -> (bool, Expr) {
    match expr {
        Expr::AggregateFunction(ref fun) => {
            let fn_name = fun.func_def.name().to_lowercase();
            if fun.distinct && WORKSPACE_ROOT_LOCK.get_or_init(|| vec!["min".to_string(), "max".to_string()]).contains(&fn_name) {
                (true, Expr::AggregateFunction(AggregateFunction{
                    func_def:fun.func_def.clone(), 
                    args:fun.args.clone(), 
                    distinct:false, 
                    filter:fun.filter.clone(),
                    order_by:fun.order_by.clone(),
                    null_treatment: fun.null_treatment
            }))
            } else {
                (false, expr)
            }
        },
        _ => (false, expr)
    }
}
impl OptimizerRule for EliminateDistinct {
    fn try_optimize(
        &self,
        _plan: &LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Option<LogicalPlan>> {
        internal_err!("Should have called EliminateDistinct::rewrite")
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Aggregate(agg) => {
                let mut aggr_expr = vec![];
                let mut transformed = false;
                for expr in agg.aggr_expr {
                    let rewrite_result = rewrite_aggr_expr(expr);
                    transformed = transformed || rewrite_result.0;
                    aggr_expr.push(rewrite_result.1);
                }

                println!("Transformed yes {}", transformed);
                let transformed = if transformed {
                    Transformed::yes
                } else {
                    Transformed::no
                };
                Aggregate::try_new(agg.input, agg.group_expr, aggr_expr)
                    .map(|f| transformed(LogicalPlan::Aggregate(f)))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
    fn name(&self) -> &str {
        "eliminate_distinct"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::*;
    use datafusion_expr::{col, logical_plan::builder::LogicalPlanBuilder};
    use datafusion_expr::AggregateExt;
    use datafusion_expr::test::function_stub::min;
    use std::sync::Arc;

    fn assert_optimized_plan_eq(plan: LogicalPlan, expected: &str) -> Result<()> {
        crate::test::assert_optimized_plan_eq(
            Arc::new(EliminateDistinct::new()),
            plan,
            expected,
        )
    }

    #[test]
    fn eliminate_distinct_from_min_expr() -> Result<()> {
        let table_scan = test_table_scan().unwrap();
        let aggr_expr = min(col("b")).distinct().build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![aggr_expr])?
            .build()?;
        let expected = "Limit: skip=5, fetch=10\
        \n  Sort: test.a, test.b, test.c\
        \n    TableScan: test";
        assert_optimized_plan_eq(plan, expected)
    }
}

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

//! [`Max`] and [`MaxAccumulator`] accumulator for the `max` function

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

//! Defines `MAX`  aggregate accumulators

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal128Array,
    Decimal256Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, LargeBinaryArray, LargeStringArray, StringArray, Time32MillisecondArray,
    Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::compute;
use arrow::datatypes::{
    DataType, Decimal128Type, Decimal256Type, Float32Type, Float64Type, Int16Type,
    Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use datafusion_common::{downcast_value, internal_err, DataFusionError, Result};
use datafusion_physical_expr_common::aggregate::groups_accumulator::prim_op::PrimitiveGroupsAccumulator;
use std::fmt::Debug;

use arrow::datatypes::{
    Date32Type, Date64Type, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
};

use arrow::datatypes::i256;

use datafusion_common::ScalarValue;
use datafusion_expr::GroupsAccumulator;
use datafusion_expr::{
    function::AccumulatorArgs, Accumulator, AggregateUDFImpl, Signature, Volatility,
};

// min/max of two non-string scalar values.
macro_rules! typed_min_max {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident $(, $EXTRA_ARGS:ident)*) => {{
        ScalarValue::$SCALAR(
            match ($VALUE, $DELTA) {
                (None, None) => None,
                (Some(a), None) => Some(*a),
                (None, Some(b)) => Some(*b),
                (Some(a), Some(b)) => Some((*a).$OP(*b)),
            },
            $($EXTRA_ARGS.clone()),*
        )
    }};
}

macro_rules! typed_min_max_float {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident) => {{
        ScalarValue::$SCALAR(match ($VALUE, $DELTA) {
            (None, None) => None,
            (Some(a), None) => Some(*a),
            (None, Some(b)) => Some(*b),
            (Some(a), Some(b)) => match a.total_cmp(b) {
                choose_min_max!($OP) => Some(*b),
                _ => Some(*a),
            },
        })
    }};
}

// Statically-typed version of min/max(array) -> ScalarValue for binay types.
macro_rules! typed_min_max_batch_binary {
    ($VALUES:expr, $ARRAYTYPE:ident, $SCALAR:ident, $OP:ident) => {{
        let array = downcast_value!($VALUES, $ARRAYTYPE);
        let value = compute::$OP(array);
        let value = value.and_then(|e| Some(e.to_vec()));
        ScalarValue::$SCALAR(value)
    }};
}

// Statically-typed version of min/max(array) -> ScalarValue for non-string types.
macro_rules! typed_min_max_batch {
    ($VALUES:expr, $ARRAYTYPE:ident, $SCALAR:ident, $OP:ident $(, $EXTRA_ARGS:ident)*) => {{
        let array = downcast_value!($VALUES, $ARRAYTYPE);
        let value = compute::$OP(array);
        ScalarValue::$SCALAR(value, $($EXTRA_ARGS.clone()),*)
    }};
}
// min/max of two scalar string values.
macro_rules! typed_min_max_string {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident) => {{
        ScalarValue::$SCALAR(match ($VALUE, $DELTA) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (Some(a), Some(b)) => Some((a).$OP(b).clone()),
        })
    }};
}

// Statically-typed version of min/max(array) -> ScalarValue for string types.
macro_rules! typed_min_max_batch_string {
    ($VALUES:expr, $ARRAYTYPE:ident, $SCALAR:ident, $OP:ident) => {{
        let array = downcast_value!($VALUES, $ARRAYTYPE);
        let value = compute::$OP(array);
        let value = value.and_then(|e| Some(e.to_string()));
        ScalarValue::$SCALAR(value)
    }};
}

macro_rules! min_max {
    ($VALUE:expr, $DELTA:expr, $OP:ident) => {{
        Ok(match ($VALUE, $DELTA) {
            (
                lhs @ ScalarValue::Decimal128(lhsv, lhsp, lhss),
                rhs @ ScalarValue::Decimal128(rhsv, rhsp, rhss)
            ) => {
                if lhsp.eq(rhsp) && lhss.eq(rhss) {
                    typed_min_max!(lhsv, rhsv, Decimal128, $OP, lhsp, lhss)
                } else {
                    return internal_err!(
                    "MIN/MAX is not expected to receive scalars of incompatible types {:?}",
                    (lhs, rhs)
                );
                }
            }
            (
                lhs @ ScalarValue::Decimal256(lhsv, lhsp, lhss),
                rhs @ ScalarValue::Decimal256(rhsv, rhsp, rhss)
            ) => {
                if lhsp.eq(rhsp) && lhss.eq(rhss) {
                    typed_min_max!(lhsv, rhsv, Decimal256, $OP, lhsp, lhss)
                } else {
                    return internal_err!(
                    "MIN/MAX is not expected to receive scalars of incompatible types {:?}",
                    (lhs, rhs)
                );
                }
            }
            (ScalarValue::Boolean(lhs), ScalarValue::Boolean(rhs)) => {
                typed_min_max!(lhs, rhs, Boolean, $OP)
            }
            (ScalarValue::Float64(lhs), ScalarValue::Float64(rhs)) => {
                typed_min_max_float!(lhs, rhs, Float64, $OP)
            }
            (ScalarValue::Float32(lhs), ScalarValue::Float32(rhs)) => {
                typed_min_max_float!(lhs, rhs, Float32, $OP)
            }
            (ScalarValue::UInt64(lhs), ScalarValue::UInt64(rhs)) => {
                typed_min_max!(lhs, rhs, UInt64, $OP)
            }
            (ScalarValue::UInt32(lhs), ScalarValue::UInt32(rhs)) => {
                typed_min_max!(lhs, rhs, UInt32, $OP)
            }
            (ScalarValue::UInt16(lhs), ScalarValue::UInt16(rhs)) => {
                typed_min_max!(lhs, rhs, UInt16, $OP)
            }
            (ScalarValue::UInt8(lhs), ScalarValue::UInt8(rhs)) => {
                typed_min_max!(lhs, rhs, UInt8, $OP)
            }
            (ScalarValue::Int64(lhs), ScalarValue::Int64(rhs)) => {
                typed_min_max!(lhs, rhs, Int64, $OP)
            }
            (ScalarValue::Int32(lhs), ScalarValue::Int32(rhs)) => {
                typed_min_max!(lhs, rhs, Int32, $OP)
            }
            (ScalarValue::Int16(lhs), ScalarValue::Int16(rhs)) => {
                typed_min_max!(lhs, rhs, Int16, $OP)
            }
            (ScalarValue::Int8(lhs), ScalarValue::Int8(rhs)) => {
                typed_min_max!(lhs, rhs, Int8, $OP)
            }
            (ScalarValue::Utf8(lhs), ScalarValue::Utf8(rhs)) => {
                typed_min_max_string!(lhs, rhs, Utf8, $OP)
            }
            (ScalarValue::LargeUtf8(lhs), ScalarValue::LargeUtf8(rhs)) => {
                typed_min_max_string!(lhs, rhs, LargeUtf8, $OP)
            }
            (ScalarValue::Binary(lhs), ScalarValue::Binary(rhs)) => {
                typed_min_max_string!(lhs, rhs, Binary, $OP)
            }
            (ScalarValue::LargeBinary(lhs), ScalarValue::LargeBinary(rhs)) => {
                typed_min_max_string!(lhs, rhs, LargeBinary, $OP)
            }
            (ScalarValue::TimestampSecond(lhs, l_tz), ScalarValue::TimestampSecond(rhs, _)) => {
                typed_min_max!(lhs, rhs, TimestampSecond, $OP, l_tz)
            }
            (
                ScalarValue::TimestampMillisecond(lhs, l_tz),
                ScalarValue::TimestampMillisecond(rhs, _),
            ) => {
                typed_min_max!(lhs, rhs, TimestampMillisecond, $OP, l_tz)
            }
            (
                ScalarValue::TimestampMicrosecond(lhs, l_tz),
                ScalarValue::TimestampMicrosecond(rhs, _),
            ) => {
                typed_min_max!(lhs, rhs, TimestampMicrosecond, $OP, l_tz)
            }
            (
                ScalarValue::TimestampNanosecond(lhs, l_tz),
                ScalarValue::TimestampNanosecond(rhs, _),
            ) => {
                typed_min_max!(lhs, rhs, TimestampNanosecond, $OP, l_tz)
            }
            (
                ScalarValue::Date32(lhs),
                ScalarValue::Date32(rhs),
            ) => {
                typed_min_max!(lhs, rhs, Date32, $OP)
            }
            (
                ScalarValue::Date64(lhs),
                ScalarValue::Date64(rhs),
            ) => {
                typed_min_max!(lhs, rhs, Date64, $OP)
            }
            (
                ScalarValue::Time32Second(lhs),
                ScalarValue::Time32Second(rhs),
            ) => {
                typed_min_max!(lhs, rhs, Time32Second, $OP)
            }
            (
                ScalarValue::Time32Millisecond(lhs),
                ScalarValue::Time32Millisecond(rhs),
            ) => {
                typed_min_max!(lhs, rhs, Time32Millisecond, $OP)
            }
            (
                ScalarValue::Time64Microsecond(lhs),
                ScalarValue::Time64Microsecond(rhs),
            ) => {
                typed_min_max!(lhs, rhs, Time64Microsecond, $OP)
            }
            (
                ScalarValue::Time64Nanosecond(lhs),
                ScalarValue::Time64Nanosecond(rhs),
            ) => {
                typed_min_max!(lhs, rhs, Time64Nanosecond, $OP)
            }
            (
                ScalarValue::IntervalYearMonth(lhs),
                ScalarValue::IntervalYearMonth(rhs),
            ) => {
                typed_min_max!(lhs, rhs, IntervalYearMonth, $OP)
            }
            (
                ScalarValue::IntervalMonthDayNano(lhs),
                ScalarValue::IntervalMonthDayNano(rhs),
            ) => {
                typed_min_max!(lhs, rhs, IntervalMonthDayNano, $OP)
            }
            (
                ScalarValue::IntervalDayTime(lhs),
                ScalarValue::IntervalDayTime(rhs),
            ) => {
                typed_min_max!(lhs, rhs, IntervalDayTime, $OP)
            }
            (
                ScalarValue::IntervalYearMonth(_),
                ScalarValue::IntervalMonthDayNano(_),
            ) | (
                ScalarValue::IntervalYearMonth(_),
                ScalarValue::IntervalDayTime(_),
            ) | (
                ScalarValue::IntervalMonthDayNano(_),
                ScalarValue::IntervalDayTime(_),
            ) | (
                ScalarValue::IntervalMonthDayNano(_),
                ScalarValue::IntervalYearMonth(_),
            ) | (
                ScalarValue::IntervalDayTime(_),
                ScalarValue::IntervalYearMonth(_),
            ) | (
                ScalarValue::IntervalDayTime(_),
                ScalarValue::IntervalMonthDayNano(_),
            ) => {
                interval_min_max!($OP, $VALUE, $DELTA)
            }
                    (
                ScalarValue::DurationSecond(lhs),
                ScalarValue::DurationSecond(rhs),
            ) => {
                typed_min_max!(lhs, rhs, DurationSecond, $OP)
            }
                                (
                ScalarValue::DurationMillisecond(lhs),
                ScalarValue::DurationMillisecond(rhs),
            ) => {
                typed_min_max!(lhs, rhs, DurationMillisecond, $OP)
            }
                                (
                ScalarValue::DurationMicrosecond(lhs),
                ScalarValue::DurationMicrosecond(rhs),
            ) => {
                typed_min_max!(lhs, rhs, DurationMicrosecond, $OP)
            }
                                        (
                ScalarValue::DurationNanosecond(lhs),
                ScalarValue::DurationNanosecond(rhs),
            ) => {
                typed_min_max!(lhs, rhs, DurationNanosecond, $OP)
            }
            e => {
                return internal_err!(
                    "MIN/MAX is not expected to receive scalars of incompatible types {:?}",
                    e
                )
            }
        })
    }};
}

macro_rules! choose_min_max {
    (min) => {
        std::cmp::Ordering::Greater
    };
    (max) => {
        std::cmp::Ordering::Less
    };
}

macro_rules! interval_min_max {
    ($OP:tt, $LHS:expr, $RHS:expr) => {{
        match $LHS.partial_cmp(&$RHS) {
            Some(choose_min_max!($OP)) => $RHS.clone(),
            Some(_) => $LHS.clone(),
            None => {
                return internal_err!("Comparison error while computing interval min/max")
            }
        }
    }};
}

// Statically-typed version of min/max(array) -> ScalarValue  for non-string types.
// this is a macro to support both operations (min and max).
macro_rules! min_max_batch {
    ($VALUES:expr, $OP:ident) => {{
        match $VALUES.data_type() {
            DataType::Decimal128(precision, scale) => {
                typed_min_max_batch!(
                    $VALUES,
                    Decimal128Array,
                    Decimal128,
                    $OP,
                    precision,
                    scale
                )
            }
            DataType::Decimal256(precision, scale) => {
                typed_min_max_batch!(
                    $VALUES,
                    Decimal256Array,
                    Decimal256,
                    $OP,
                    precision,
                    scale
                )
            }
            // all types that have a natural order
            DataType::Float64 => {
                typed_min_max_batch!($VALUES, Float64Array, Float64, $OP)
            }
            DataType::Float32 => {
                typed_min_max_batch!($VALUES, Float32Array, Float32, $OP)
            }
            DataType::Int64 => typed_min_max_batch!($VALUES, Int64Array, Int64, $OP),
            DataType::Int32 => typed_min_max_batch!($VALUES, Int32Array, Int32, $OP),
            DataType::Int16 => typed_min_max_batch!($VALUES, Int16Array, Int16, $OP),
            DataType::Int8 => typed_min_max_batch!($VALUES, Int8Array, Int8, $OP),
            DataType::UInt64 => typed_min_max_batch!($VALUES, UInt64Array, UInt64, $OP),
            DataType::UInt32 => typed_min_max_batch!($VALUES, UInt32Array, UInt32, $OP),
            DataType::UInt16 => typed_min_max_batch!($VALUES, UInt16Array, UInt16, $OP),
            DataType::UInt8 => typed_min_max_batch!($VALUES, UInt8Array, UInt8, $OP),
            DataType::Timestamp(TimeUnit::Second, tz_opt) => {
                typed_min_max_batch!(
                    $VALUES,
                    TimestampSecondArray,
                    TimestampSecond,
                    $OP,
                    tz_opt
                )
            }
            DataType::Timestamp(TimeUnit::Millisecond, tz_opt) => typed_min_max_batch!(
                $VALUES,
                TimestampMillisecondArray,
                TimestampMillisecond,
                $OP,
                tz_opt
            ),
            DataType::Timestamp(TimeUnit::Microsecond, tz_opt) => typed_min_max_batch!(
                $VALUES,
                TimestampMicrosecondArray,
                TimestampMicrosecond,
                $OP,
                tz_opt
            ),
            DataType::Timestamp(TimeUnit::Nanosecond, tz_opt) => typed_min_max_batch!(
                $VALUES,
                TimestampNanosecondArray,
                TimestampNanosecond,
                $OP,
                tz_opt
            ),
            DataType::Date32 => typed_min_max_batch!($VALUES, Date32Array, Date32, $OP),
            DataType::Date64 => typed_min_max_batch!($VALUES, Date64Array, Date64, $OP),
            DataType::Time32(TimeUnit::Second) => {
                typed_min_max_batch!($VALUES, Time32SecondArray, Time32Second, $OP)
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                typed_min_max_batch!(
                    $VALUES,
                    Time32MillisecondArray,
                    Time32Millisecond,
                    $OP
                )
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                typed_min_max_batch!(
                    $VALUES,
                    Time64MicrosecondArray,
                    Time64Microsecond,
                    $OP
                )
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                typed_min_max_batch!(
                    $VALUES,
                    Time64NanosecondArray,
                    Time64Nanosecond,
                    $OP
                )
            }
            other => {
                // This should have been handled before
                return internal_err!(
                    "Min/Max accumulator not implemented for type {:?}",
                    other
                );
            }
        }
    }};
}

/// dynamically-typed max(array) -> ScalarValue
fn max_batch(values: &ArrayRef) -> Result<ScalarValue> {
    Ok(match values.data_type() {
        DataType::Utf8 => {
            typed_min_max_batch_string!(values, StringArray, Utf8, max_string)
        }
        DataType::LargeUtf8 => {
            typed_min_max_batch_string!(values, LargeStringArray, LargeUtf8, max_string)
        }
        DataType::Boolean => {
            typed_min_max_batch!(values, BooleanArray, Boolean, max_boolean)
        }
        DataType::Binary => {
            typed_min_max_batch_binary!(&values, BinaryArray, Binary, max_binary)
        }
        DataType::LargeBinary => {
            typed_min_max_batch_binary!(
                &values,
                LargeBinaryArray,
                LargeBinary,
                max_binary
            )
        }
        _ => min_max_batch!(values, max),
    })
}

// min/max of two non-string scalar values.
macro_rules! typed_min_max {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident $(, $EXTRA_ARGS:ident)*) => {{
        ScalarValue::$SCALAR(
            match ($VALUE, $DELTA) {
                (None, None) => None,
                (Some(a), None) => Some(*a),
                (None, Some(b)) => Some(*b),
                (Some(a), Some(b)) => Some((*a).$OP(*b)),
            },
            $($EXTRA_ARGS.clone()),*
        )
    }};
}

make_udaf_expr_and_func!(
    Max,
    max,
    expression,
    "Returns the maximum of a group of values.",
    max_udaf
);

make_udaf_expr_and_func!(
    Min,
    min,
    expression,
    "Returns the minimum of a group of values.",
    min_udaf
);

fn min_max_aggregate_data_type(input_type: DataType) -> DataType {
    if let DataType::Dictionary(_, value_type) = input_type {
        *value_type
    } else {
        input_type
    }
}

#[derive(Debug)]
pub struct Max {
    signature: Signature,
    aliases: Vec<String>,
}

impl Max {
    pub fn new() -> Self {
        Self {
            signature: Signature::numeric(1, Volatility::Immutable),
            aliases: vec!["max".to_owned()],
        }
    }
}

impl Default for Max {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateUDFImpl for Max {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "MAX"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(min_max_aggregate_data_type(arg_types[0].clone()))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(MaxAccumulator::try_new(acc_args.data_type)?))
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn groups_accumulator_supported(&self, _args: AccumulatorArgs) -> bool {
        true
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        use DataType::*;
        use TimeUnit::*;
        let data_type = args.data_type;
        macro_rules! helper {
            ($NATIVE:ident, $PRIMTYPE:ident) => {{
                Ok(Box::new(
                    PrimitiveGroupsAccumulator::<$PRIMTYPE, _>::new(
                        data_type,
                        |cur, new| {
                            if *cur < new {
                                *cur = new
                            }
                        },
                    )
                    .with_starting_value($NATIVE::MIN),
                ))
            }};
        }

        match args.data_type {
            Int8 => helper!(i8, Int8Type),
            Int16 => helper!(i16, Int16Type),
            Int32 => helper!(i32, Int32Type),
            Int64 => helper!(i64, Int64Type),
            UInt8 => helper!(u8, UInt8Type),
            UInt16 => helper!(u16, UInt16Type),
            UInt32 => helper!(u32, UInt32Type),
            UInt64 => helper!(u64, UInt64Type),
            Float32 => {
                helper!(f32, Float32Type)
            }
            Float64 => {
                helper!(f64, Float64Type)
            }
            Date32 => helper!(i32, Date32Type),
            Date64 => helper!(i64, Date64Type),
            Time32(Second) => {
                helper!(i32, Time32SecondType)
            }
            Time32(Millisecond) => {
                helper!(i32, Time32MillisecondType)
            }
            Time64(Microsecond) => {
                helper!(i64, Time64MicrosecondType)
            }
            Time64(Nanosecond) => {
                helper!(i64, Time64NanosecondType)
            }
            Timestamp(Second, _) => {
                helper!(i64, TimestampSecondType)
            }
            Timestamp(Millisecond, _) => {
                helper!(i64, TimestampMillisecondType)
            }
            Timestamp(Microsecond, _) => {
                helper!(i64, TimestampMicrosecondType)
            }
            Timestamp(Nanosecond, _) => {
                helper!(i64, TimestampNanosecondType)
            }
            Decimal128(_, _) => {
                helper!(i128, Decimal128Type)
            }
            Decimal256(_, _) => {
                helper!(i256, Decimal256Type)
            }

            // It would be nice to have a fast implementation for Strings as well
            // https://github.com/apache/datafusion/issues/6906

            // This is only reached if groups_accumulator_supported is out of sync
            _ => internal_err!("GroupsAccumulator not supported for max({})", data_type),
        }
    }
}

/// An accumulator to compute the maximum value
#[derive(Debug)]
pub struct MaxAccumulator {
    max: ScalarValue,
}

impl MaxAccumulator {
    /// new max accumulator
    pub fn try_new(datatype: &DataType) -> Result<Self> {
        Ok(Self {
            max: ScalarValue::try_from(datatype)?,
        })
    }
}

impl Accumulator for MaxAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = &values[0];
        let delta = &max_batch(values)?;
        let new_max: Result<ScalarValue, DataFusionError> =
            min_max!(&self.max, delta, max);
        self.max = new_max?;
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(self.max.clone())
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) - std::mem::size_of_val(&self.max) + self.max.size()
    }
}

#[derive(Debug)]
pub struct Min {
    signature: Signature,
    aliases: Vec<String>,
}

impl Min {
    pub fn new() -> Self {
        Self {
            signature: Signature::numeric(1, Volatility::Immutable),
            aliases: vec!["min".to_owned()],
        }
    }
}
impl Default for Min {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateUDFImpl for Min {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "min"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(min_max_aggregate_data_type(arg_types[0].clone()))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(MinAccumulator::try_new(acc_args.data_type)?))
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn groups_accumulator_supported(&self, _args: AccumulatorArgs) -> bool {
        true
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        use DataType::*;
        use TimeUnit::*;
        let data_type = args.data_type;
        macro_rules! helper {
            ($NATIVE:ident, $PRIMTYPE:ident) => {{
                Ok(Box::new(
                    PrimitiveGroupsAccumulator::<$PRIMTYPE, _>::new(
                        data_type,
                        |cur, new| {
                            if *cur > new {
                                *cur = new
                            }
                        },
                    )
                    .with_starting_value($NATIVE::MAX),
                ))
            }};
        }

        match args.data_type {
            Int8 => helper!(i8, Int8Type),
            Int16 => helper!(i16, Int16Type),
            Int32 => helper!(i32, Int32Type),
            Int64 => helper!(i64, Int64Type),
            UInt8 => helper!(u8, UInt8Type),
            UInt16 => helper!(u16, UInt16Type),
            UInt32 => helper!(u32, UInt32Type),
            UInt64 => helper!(u64, UInt64Type),
            Float32 => {
                helper!(f32, Float32Type)
            }
            Float64 => {
                helper!(f64, Float64Type)
            }
            Date32 => helper!(i32, Date32Type),
            Date64 => helper!(i64, Date64Type),
            Time32(Second) => {
                helper!(i32, Time32SecondType)
            }
            Time32(Millisecond) => {
                helper!(i32, Time32MillisecondType)
            }
            Time64(Microsecond) => {
                helper!(i64, Time64MicrosecondType)
            }
            Time64(Nanosecond) => {
                helper!(i64, Time64NanosecondType)
            }
            Timestamp(Second, _) => {
                helper!(i64, TimestampSecondType)
            }
            Timestamp(Millisecond, _) => {
                helper!(i64, TimestampMillisecondType)
            }
            Timestamp(Microsecond, _) => {
                helper!(i64, TimestampMicrosecondType)
            }
            Timestamp(Nanosecond, _) => {
                helper!(i64, TimestampNanosecondType)
            }
            Decimal128(_, _) => {
                helper!(i128, Decimal128Type)
            }
            Decimal256(_, _) => {
                helper!(i256, Decimal256Type)
            }

            // It would be nice to have a fast implementation for Strings as well
            // https://github.com/apache/datafusion/issues/6906

            // This is only reached if groups_accumulator_supported is out of sync
            _ => internal_err!("GroupsAccumulator not supported for min({})", data_type),
        }
    }
}
/// An accumulator to compute the minimum value
#[derive(Debug)]
pub struct MinAccumulator {
    min: ScalarValue,
}

impl MinAccumulator {
    /// new max accumulator
    pub fn try_new(datatype: &DataType) -> Result<Self> {
        Ok(Self {
            min: ScalarValue::try_from(datatype)?,
        })
    }
}

impl Accumulator for MinAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = &values[0];
        let delta = &max_batch(values)?;
        let new_min: Result<ScalarValue, DataFusionError> =
            min_max!(&self.min, delta, min);
        self.min = new_min?;
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(self.min.clone())
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) - std::mem::size_of_val(&self.min) + self.min.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn float_min_max_with_nans() {
        let pos_nan = f32::NAN;
        let zero = 0_f32;
        let neg_inf = f32::NEG_INFINITY;

        let check = |acc: &mut dyn Accumulator, values: &[&[f32]], expected: f32| {
            for batch in values.iter() {
                let batch =
                    Arc::new(Float32Array::from_iter_values(batch.iter().copied()));
                acc.update_batch(&[batch]).unwrap();
            }
            let result = acc.evaluate().unwrap();
            assert_eq!(result, ScalarValue::Float32(Some(expected)));
        };

        // This test checks both comparison between batches (which uses the min_max macro
        // defined above) and within a batch (which uses the arrow min/max compute function
        // and verifies both respect the total order comparison for floats)

        let min = || MinAccumulator::try_new(&DataType::Float32).unwrap();
        let max = || MaxAccumulator::try_new(&DataType::Float32).unwrap();

        check(&mut min(), &[&[zero], &[pos_nan]], zero);
        check(&mut min(), &[&[zero, pos_nan]], zero);
        check(&mut min(), &[&[zero], &[neg_inf]], neg_inf);
        check(&mut min(), &[&[zero, neg_inf]], neg_inf);
        check(&mut max(), &[&[zero], &[pos_nan]], pos_nan);
        check(&mut max(), &[&[zero, pos_nan]], pos_nan);
        check(&mut max(), &[&[zero], &[neg_inf]], zero);
        check(&mut max(), &[&[zero, neg_inf]], zero);
    }
}

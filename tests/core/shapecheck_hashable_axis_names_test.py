# Copyright 2026 The Penzai Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for hashable named axes in shape checking."""

from absl.testing import absltest
import jax
from penzai import pz


class ShapecheckHashableAxisNamesTest(absltest.TestCase):

  def test_repeated_named_unpack_accepts_hashable_axis_name(self):
    temp_axis = pz.nx.TmpPosAxisMarker()

    match = pz.chk.check_structure(
        value={
            "a": pz.chk.ArraySpec(named_shape={"feature": 2, temp_axis: 3}),
            "b": pz.chk.ArraySpec(named_shape={"feature": 4, temp_axis: 3}),
        },
        pattern={
            "a": pz.chk.ArraySpec(
                named_shape={"feature": 2, **pz.chk.var("batch_axes")}
            ),
            "b": pz.chk.ArraySpec(
                named_shape={"feature": 4, **pz.chk.var("batch_axes")}
            ),
        },
    )

    self.assertEqual(match["batch_axes"], {temp_axis: 3})

  def test_linear_accepts_temporary_positional_axis_marker(self):
    temp_axis = pz.nx.TmpPosAxisMarker()
    layer = pz.nn.Linear.from_config(
        "my_layer",
        jax.random.key(0),
        input_axes={"in_axis": 2},
        output_axes={"out_axis": 3},
    )

    result = layer(pz.nx.zeros({"in_axis": 2, temp_axis: 3}))

    self.assertEqual(result.named_shape, {temp_axis: 3, "out_axis": 3})


if __name__ == "__main__":
  absltest.main()

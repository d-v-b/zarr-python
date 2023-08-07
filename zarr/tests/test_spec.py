from zarr.hierarchy import ArraySpecV2
from zarr.hierarchy import GroupSpecV2
from zarr import MemoryStore
import pytest
import numpy as np
from numcodecs import GZip, FixedScaleOffset, Blosc

argnames = (
    "shape, dtype, chunks, compressor, filters, dimension_separator, order, fill_value, attrs"
)
argvalues = (
    ((10,), np.uint8, (1,), None, None, ".", "C", 0, {"foo": 10}),
    (
        (10, 10),
        np.float32,
        (5, 5),
        GZip(-1),
        [FixedScaleOffset(0, 1, np.float32)],
        "/",
        "C",
        2,
        {"foo": 10},
    ),
    ((10, 20, 30), np.int16, (1, 2, 3), Blosc(cname="zstd"), None, ".", "F", 1, {"foo": 10}),
)


@pytest.mark.parametrize(argnames, argvalues)
def test_arrayspecv2(
    shape, dtype, chunks, compressor, filters, dimension_separator, order, fill_value, attrs
):

    store = MemoryStore()

    array_spec = ArraySpecV2(
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
        filters=filters,
        dimension_separator=dimension_separator,
        order=order,
        fill_value=fill_value,
        attrs=attrs,
    )

    array = array_spec.to_storage(store, path="foo")
    assert array._version == 2
    assert array.path == "foo"
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.chunks == chunks
    assert array.compressor == compressor
    assert array.filters == filters
    assert array._dimension_separator == dimension_separator
    assert array.order == order
    assert array.fill_value == fill_value
    assert array.attrs == attrs

    array_spec2 = ArraySpecV2.from_storage(array)
    assert array_spec2 == array_spec


def test_groupspecv2():
    store = MemoryStore()
    sub_sub_groupspec = GroupSpecV2(attrs={"group": True}, nodes={})
    sub_groupspec = GroupSpecV2(
        attrs={"bar": False, "baz": True}, nodes={"sub_group": sub_sub_groupspec}
    )
    sub_arrayspec = ArraySpecV2(
        attrs={"array": True},
        shape=(1,),
        dtype=np.uint8,
        chunks=(1,),
        compressor=GZip(-1),
        filters=None,
        fill_value=0,
        order="C",
        dimension_separator="/",
    )
    group_spec = GroupSpecV2(
        attrs={"foo": [1, 2, 3]}, nodes={"sub_group": sub_groupspec, "sub_array": sub_arrayspec}
    )
    group = group_spec.to_storage(store, path="")
    assert group._version == 2
    assert GroupSpecV2.from_storage(group) == group_spec
    assert group.path == ""
    assert group.attrs == group_spec.attrs
    sub_group = group["sub_group"]
    assert GroupSpecV2.from_storage(sub_group) == sub_groupspec
    sub_array = group["sub_array"]
    assert ArraySpecV2.from_storage(sub_array) == sub_arrayspec
    sub_sub_group = group["sub_group"]["sub_group"]
    assert GroupSpecV2.from_storage(sub_sub_group) == sub_sub_groupspec

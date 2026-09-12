from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Self,
    SupportsIndex,
    SupportsInt,
    TypeGuard,
    overload,
)

import numpy as np

from zarr.core.dtype.common import (
    DTypeConfig_V2,
    DTypeJSON,
    HasEndianness,
    HasItemSize,
    check_dtype_spec_v2,
)
from zarr.core.dtype.npy.common import (
    check_json_int,
    check_json_intish_float,
    check_json_intish_str,
    endianness_to_numpy_str,
    get_endianness_from_numpy_dtype,
)
from zarr.core.dtype.wrapper import TBaseDType, ZDType
from zarr.errors import DataTypeValidationError

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

_NumpyIntDType = (
    np.dtypes.Int8DType
    | np.dtypes.Int16DType
    | np.dtypes.Int32DType
    | np.dtypes.Int64DType
    | np.dtypes.UInt8DType
    | np.dtypes.UInt16DType
    | np.dtypes.UInt32DType
    | np.dtypes.UInt64DType
)
_NumpyIntScalar = (
    np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
)

IntLike = SupportsInt | SupportsIndex | bytes | str


@dataclass(frozen=True)
class BaseInt[
    DType: _NumpyIntDType,
    Scalar: np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64,
](ZDType[DType, Scalar], HasItemSize):
    """
    A base class for integer data types in Zarr.

    This class provides methods for serialization and deserialization of integer types
    in both Zarr v2 and v3 formats, as well as methods for checking and casting scalars.

    Subclasses provide the concrete ``dtype_cls``, ``_zarr_v3_name``, ``_zarr_v2_names``,
    and ``item_size`` attributes. Multi-byte integer types additionally inherit from
    [`HasEndianness`][zarr.core.dtype.common.HasEndianness]; the byte-order logic in this
    base class is gated on that so that the single-byte ``Int8``/``UInt8`` types (which have
    no meaningful byte order) reuse the same implementation.
    """

    _zarr_v2_names: ClassVar[tuple[str, ...]]

    # Single-byte int types (Int8/UInt8) have no meaningful byte order, so they do not mix in
    # HasEndianness. Multi-byte types do; this flag (set True by HasEndianness subclasses) lets the
    # shared native-dtype conversion below branch without tripping mypy's issubclass narrowing.
    _has_endianness: ClassVar[bool] = False

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        """
        Create an instance of this data type from a native NumPy dtype.

        Parameters
        ----------
        dtype : TBaseDType
            The native NumPy dtype.

        Returns
        -------
        Self
            An instance of this data type.

        Raises
        ------
        DataTypeValidationError
            If the input dtype is not a valid representation of this data type.
        """
        if cls._check_native_dtype(dtype):
            kwargs: dict[str, object] = {}
            if cls._has_endianness:
                kwargs["endianness"] = get_endianness_from_numpy_dtype(dtype)
            return cls(**kwargs)
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> DType:
        """
        Convert this data type to a native NumPy dtype.

        Returns
        -------
        DType
            The native NumPy dtype.
        """
        if isinstance(self, HasEndianness):
            byte_order = endianness_to_numpy_str(self.endianness)
            # numpy 2.x stub: newbyteorder widens to base dtype, runtime preserves the subclass
            return self.dtype_cls().newbyteorder(byte_order)  # type: ignore[no-any-return,call-overload]
        return self.dtype_cls()  # type: ignore[no-any-return,call-overload]

    @classmethod
    def _check_json_v2(cls, data: object) -> TypeGuard[DTypeConfig_V2[str, None]]:
        """
        Check that the input is a valid JSON representation of this integer data type in Zarr V2.

        This method verifies that the provided data matches the expected Zarr V2 representation
        for this data type. The input data must be a mapping that contains a "name" key that is
        one of the strings from cls._zarr_v2_names and an "object_codec_id" key that is None.

        Parameters
        ----------
        data : object
            The JSON data to check.

        Returns
        -------
        TypeGuard[DTypeConfig_V2[str, None]]
            True if the input is a valid representation of this class in Zarr V2,
            False otherwise.
        """

        return (
            check_dtype_spec_v2(data)
            and data["name"] in cls._zarr_v2_names
            and data["object_codec_id"] is None
        )

    @classmethod
    def _check_json_v3(cls, data: object) -> TypeGuard[str]:
        """
        Check that the input is a valid JSON representation of this class in Zarr V3.

        Parameters
        ----------
        data : object
            The JSON data to check.

        Returns
        -------
        TypeGuard[str]
            True if the input is a valid representation of this class in Zarr v3,
            False otherwise.
        """
        return data == cls._zarr_v3_name

    @classmethod
    def _from_json_v2(cls, data: DTypeJSON) -> Self:
        """
        Create an instance of this data type from Zarr V2-flavored JSON.

        Parameters
        ----------
        data : DTypeJSON
            The JSON data.

        Returns
        -------
        Self
            An instance of this data type.

        Raises
        ------
        DataTypeValidationError
            If the input JSON is not a valid representation of this class.
        """
        if cls._check_json_v2(data):
            # Going via NumPy ensures that we get the endianness correct without
            # annoying string parsing.
            name = data["name"]
            return cls.from_native_dtype(np.dtype(name))
        msg = (
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}, "
            f"expected one of the strings {cls._zarr_v2_names!r}."
        )
        raise DataTypeValidationError(msg)

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> Self:
        """
        Create an instance of this data type from Zarr V3-flavored JSON.

        Parameters
        ----------
        data : DTypeJSON
            The JSON data.

        Returns
        -------
        Self
            An instance of this data type.

        Raises
        ------
        DataTypeValidationError
            If the input JSON is not a valid representation of this class.
        """
        if cls._check_json_v3(data):
            return cls()
        msg = (
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}, "
            f"expected the string {cls._zarr_v3_name!r}"
        )
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> DTypeConfig_V2[str, None]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> str: ...

    def to_json(self, zarr_format: ZarrFormat) -> DTypeConfig_V2[str, None] | str:
        """
        Convert the data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The Zarr format version.

        Returns
        -------
        DTypeConfig_V2[str, None] or str
            The JSON-serializable representation of the data type.

        Raises
        ------
        ValueError
            If the zarr_format is not 2 or 3.
        """
        if zarr_format == 2:
            return {"name": self.to_native_dtype().str, "object_codec_id": None}
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def _check_scalar(self, data: object) -> TypeGuard[IntLike]:
        """
        Check if the input object is of an IntLike type.

        This method verifies whether the provided data can be considered as an integer-like
        value, which includes objects supporting integer conversion.

        Parameters
        ----------
        data : object
            The data to check.

        Returns
        -------
        TypeGuard[IntLike]
            True if the data is IntLike, False otherwise.
        """

        return isinstance(data, IntLike)

    def _cast_scalar_unchecked(self, data: IntLike) -> Scalar:
        """
        Casts a given scalar value to the native integer scalar type without type checking.

        Parameters
        ----------
        data : IntLike
            The scalar value to cast.

        Returns
        -------
        Scalar
            The casted integer scalar of the native dtype.
        """

        return self.to_native_dtype().type(data)  # type: ignore[return-value]

    def cast_scalar(self, data: object) -> Scalar:
        """
        Attempt to cast a given object to a NumPy integer scalar.

        Parameters
        ----------
        data : object
            The data to be cast to a NumPy integer scalar.

        Returns
        -------
        Scalar
            The data cast as a NumPy integer scalar.

        Raises
        ------
        TypeError
            If the data cannot be converted to a NumPy integer scalar.
        """

        if self._check_scalar(data):
            return self._cast_scalar_unchecked(data)
        msg = (
            f"Cannot convert object {data!r} with type {type(data)} to a scalar compatible with the "
            f"data type {self}."
        )
        raise TypeError(msg)

    def default_scalar(self) -> Scalar:
        """
        Get the default value, which is 0 cast to this dtype.

        Returns
        -------
        Scalar
            The default value.
        """
        return self._cast_scalar_unchecked(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> Scalar:
        """
        Read a JSON-serializable value as a NumPy int scalar.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value.
        zarr_format : ZarrFormat
            The Zarr format version.

        Returns
        -------
        Scalar
            The NumPy int scalar.

        Raises
        ------
        TypeError
            If the input is not a valid integer type.
        """
        if check_json_int(data):
            return self._cast_scalar_unchecked(data)
        if check_json_intish_float(data):
            return self._cast_scalar_unchecked(int(data))

        if check_json_intish_str(data):
            return self._cast_scalar_unchecked(int(data))

        raise TypeError(f"Invalid type: {data}. Expected an integer.")

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> int:
        """
        Convert an object to a JSON serializable scalar. For the integer data types,
        the JSON form is a plain integer.

        Parameters
        ----------
        data : object
            The value to convert.
        zarr_format : ZarrFormat
            The Zarr format version.

        Returns
        -------
        int
            The JSON-serializable form of the scalar.
        """
        return int(self.cast_scalar(data))


@dataclass(frozen=True, kw_only=True)
class Int8(BaseInt[np.dtypes.Int8DType, np.int8]):
    """
    A Zarr data type for arrays containing 8-bit signed integers.

    Wraps the [`np.dtypes.Int8DType`][numpy.dtypes.Int8DType] data type. Scalars for this data type are
    instances of [`np.int8`][numpy.int8].

    Attributes
    ----------
    dtype_cls : np.dtypes.Int8DType
        The class of the underlying NumPy dtype.

    References
    ----------
    This class implements the 8-bit signed integer data type defined in Zarr V2 and V3.

    See the [Zarr V2](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding) and [Zarr V3](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst) specification documents for details.
    """

    dtype_cls = np.dtypes.Int8DType
    _zarr_v3_name: ClassVar[Literal["int8"]] = "int8"
    _zarr_v2_names: ClassVar[tuple[Literal["|i1"]]] = ("|i1",)

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return 1


@dataclass(frozen=True, kw_only=True)
class UInt8(BaseInt[np.dtypes.UInt8DType, np.uint8]):
    """
    A Zarr data type for arrays containing 8-bit unsigned integers.

    Wraps the [`np.dtypes.UInt8DType`][numpy.dtypes.UInt8DType] data type. Scalars for this data type are instances of [`np.uint8`][numpy.uint8].

    Attributes
    ----------
    dtype_cls : np.dtypes.UInt8DType
        The class of the underlying NumPy dtype.

    References
    ----------
    This class implements the 8-bit unsigned integer data type defined in Zarr V2 and V3.

    See the [Zarr V2](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding) and [Zarr V3](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst) specification documents for details.
    """

    dtype_cls = np.dtypes.UInt8DType
    _zarr_v3_name: ClassVar[Literal["uint8"]] = "uint8"
    _zarr_v2_names: ClassVar[tuple[Literal["|u1"]]] = ("|u1",)

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return 1


@dataclass(frozen=True, kw_only=True)
class Int16(BaseInt[np.dtypes.Int16DType, np.int16], HasEndianness):
    """
    A Zarr data type for arrays containing 16-bit signed integers.

    Wraps the [`np.dtypes.Int16DType`][numpy.dtypes.Int16DType] data type. Scalars for this data type are instances of
    [`np.int16`][numpy.int16].

    Attributes
    ----------
    dtype_cls : np.dtypes.Int16DType
        The class of the underlying NumPy dtype.

    References
    ----------
    This class implements the 16-bit signed integer data type defined in Zarr V2 and V3.

    See the [Zarr V2](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding) and [Zarr V3](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst) specification documents for details.
    """

    dtype_cls = np.dtypes.Int16DType
    _zarr_v3_name: ClassVar[Literal["int16"]] = "int16"
    _zarr_v2_names: ClassVar[tuple[Literal[">i2"], Literal["<i2"]]] = (">i2", "<i2")
    _has_endianness: ClassVar[bool] = True

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return 2


@dataclass(frozen=True, kw_only=True)
class UInt16(BaseInt[np.dtypes.UInt16DType, np.uint16], HasEndianness):
    """
    A Zarr data type for arrays containing 16-bit unsigned integers.

    Wraps the [`np.dtypes.UInt16DType`][numpy.dtypes.UInt16DType] data type. Scalars for this data type are instances of
    [`np.uint16`][numpy.uint16].

    Attributes
    ----------
    dtype_cls : np.dtypes.UInt16DType
        The class of the underlying NumPy dtype.

    References
    ----------
    This class implements the unsigned 16-bit unsigned integer data type defined in Zarr V2 and V3.

    See the [Zarr V2](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding) and [Zarr V3](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst) specification documents for details.
    """

    dtype_cls = np.dtypes.UInt16DType
    _zarr_v3_name: ClassVar[Literal["uint16"]] = "uint16"
    _zarr_v2_names: ClassVar[tuple[Literal[">u2"], Literal["<u2"]]] = (">u2", "<u2")
    _has_endianness: ClassVar[bool] = True

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return 2


@dataclass(frozen=True, kw_only=True)
class Int32(BaseInt[np.dtypes.Int32DType, np.int32], HasEndianness):
    """
    A Zarr data type for arrays containing 32-bit signed integers.

    Wraps the [`np.dtypes.Int32DType`][numpy.dtypes.Int32DType] data type. Scalars for this data type are instances of
    [`np.int32`][numpy.int32].

    Attributes
    ----------
    dtype_cls : np.dtypes.Int32DType
        The class of the underlying NumPy dtype.

    References
    ----------
    This class implements the 32-bit signed integer data type defined in Zarr V2 and V3.

    See the [Zarr V2](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding) and [Zarr V3](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst) specification documents for details.
    """

    dtype_cls = np.dtypes.Int32DType
    _zarr_v3_name: ClassVar[Literal["int32"]] = "int32"
    _zarr_v2_names: ClassVar[tuple[Literal[">i4"], Literal["<i4"]]] = (">i4", "<i4")
    _has_endianness: ClassVar[bool] = True

    @classmethod
    def _check_native_dtype(cls: type[Self], dtype: TBaseDType) -> TypeGuard[np.dtypes.Int32DType]:
        """
        A type guard that checks if the input is assignable to the type of ``cls.dtype_class``

        This method is overridden for this particular data type because of a Windows-specific issue
        where np.dtype('i') creates an instance of ``np.dtypes.IntDType``, rather than an
        instance of ``np.dtypes.Int32DType``, even though both represent 32-bit signed integers.

        Parameters
        ----------
        dtype : TDType
            The dtype to check.

        Returns
        -------
        Bool
            True if the dtype matches, False otherwise.
        """
        return super()._check_native_dtype(dtype) or dtype == np.dtypes.Int32DType()

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return 4


@dataclass(frozen=True, kw_only=True)
class UInt32(BaseInt[np.dtypes.UInt32DType, np.uint32], HasEndianness):
    """
    A Zarr data type for arrays containing 32-bit unsigned integers.

    Wraps the [`np.dtypes.UInt32DType`][numpy.dtypes.UInt32DType] data type. Scalars for this data type are instances of
    [`np.uint32`][numpy.uint32].

    Attributes
    ----------
    dtype_cls : np.dtypes.UInt32DType
        The class of the underlying NumPy dtype.

    References
    ----------
    This class implements the 32-bit unsigned integer data type defined in Zarr V2 and V3.

    See the [Zarr V2](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding) and [Zarr V3](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst) specification documents for details.
    """

    dtype_cls = np.dtypes.UInt32DType
    _zarr_v3_name: ClassVar[Literal["uint32"]] = "uint32"
    _zarr_v2_names: ClassVar[tuple[Literal[">u4"], Literal["<u4"]]] = (">u4", "<u4")
    _has_endianness: ClassVar[bool] = True

    @classmethod
    def _check_native_dtype(cls: type[Self], dtype: TBaseDType) -> TypeGuard[np.dtypes.UInt32DType]:
        """
        A type guard that checks if the input is assignable to the type of ``cls.dtype_class``

        This method is overridden for this particular data type because of a Windows-specific issue
        where ``np.array([1], dtype=np.uint32) & 1`` creates an instance of ``np.dtypes.UIntDType``,
        rather than an instance of ``np.dtypes.UInt32DType``, even though both represent 32-bit
        unsigned integers. (In contrast to ``np.dtype('i')``, ``np.dtype('u')`` raises an error.)

        Parameters
        ----------
        dtype : TDType
            The dtype to check.

        Returns
        -------
        Bool
            True if the dtype matches, False otherwise.
        """
        return super()._check_native_dtype(dtype) or dtype == np.dtypes.UInt32DType()

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return 4


@dataclass(frozen=True, kw_only=True)
class Int64(BaseInt[np.dtypes.Int64DType, np.int64], HasEndianness):
    """
    A Zarr data type for arrays containing 64-bit signed integers.

    Wraps the [`np.dtypes.Int64DType`][numpy.dtypes.Int64DType] data type. Scalars for this data type are instances of
    [`np.int64`][numpy.int64].

    Attributes
    ----------
    dtype_cls : np.dtypes.Int64DType
        The class of the underlying NumPy dtype.

    References
    ----------
    This class implements the 64-bit signed integer data type defined in Zarr V2 and V3.

    See the [Zarr V2](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding) and [Zarr V3](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst) specification documents for details.
    """

    dtype_cls = np.dtypes.Int64DType
    _zarr_v3_name: ClassVar[Literal["int64"]] = "int64"
    _zarr_v2_names: ClassVar[tuple[Literal[">i8"], Literal["<i8"]]] = (">i8", "<i8")
    _has_endianness: ClassVar[bool] = True

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return 8


@dataclass(frozen=True, kw_only=True)
class UInt64(BaseInt[np.dtypes.UInt64DType, np.uint64], HasEndianness):
    """
    A Zarr data type for arrays containing 64-bit unsigned integers.

    Wraps the [`np.dtypes.UInt64DType`][numpy.dtypes.UInt64DType] data type. Scalars for this data type
    are instances of [`np.uint64`][numpy.uint64].

    Attributes
    ----------
    dtype_cls: np.dtypes.UInt64DType
        The class of the underlying NumPy dtype.

    References
    ----------
    This class implements the unsigned 64-bit integer data type defined in Zarr V2 and V3.

    See the [Zarr V2](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding) and [Zarr V3](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst) specification documents for details.
    """

    dtype_cls = np.dtypes.UInt64DType
    _zarr_v3_name: ClassVar[Literal["uint64"]] = "uint64"
    _zarr_v2_names: ClassVar[tuple[Literal[">u8"], Literal["<u8"]]] = (">u8", "<u8")
    _has_endianness: ClassVar[bool] = True

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return 8

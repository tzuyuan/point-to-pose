import numpy as np


CHANNEL_TYPE_TO_DTYPE = {
    0: np.int8,
    1: np.uint8,
    2: np.int16,
    3: np.uint16,
    4: np.int32,
    5: np.uint32,
    6: np.float32,
    7: np.float64,
}

DTYPE_TO_CHANNEL_TYPE = {v: k for k, v in CHANNEL_TYPE_TO_DTYPE.items()}


def pack_image_to_bytes(image: np.ndarray, channel_type: int) -> bytes:
    dtype = CHANNEL_TYPE_TO_DTYPE.get(int(channel_type))
    if dtype is None:
        raise ValueError(f"Unsupported channel type: {channel_type}")
    if image.dtype != dtype:
        raise ValueError(
            f"Image dtype {image.dtype} does not match channel type {channel_type}"
        )
    return image.tobytes()


def unpack_image_from_bytes(
    data: bytes,
    height: int,
    width: int,
    num_channels: int,
    channel_type: int,
) -> np.ndarray:
    dtype = CHANNEL_TYPE_TO_DTYPE.get(int(channel_type))
    if dtype is None:
        raise ValueError(f"Unsupported channel type: {channel_type}")

    expected_size = height * width * num_channels * np.dtype(dtype).itemsize
    if len(data) != expected_size:
        raise ValueError(
            f"Data size mismatch: expected {expected_size} bytes, got {len(data)}"
        )

    flat = np.frombuffer(data, dtype=dtype)
    if num_channels == 1:
        return flat.reshape((height, width))
    return flat.reshape((height, width, num_channels))

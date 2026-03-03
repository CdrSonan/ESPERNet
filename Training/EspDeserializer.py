import numpy as np


def deserialize_esper_audio_uncompressed(
        data: bytes,
        expected_file_standard: int,
        expected_n_voiced: int,
        expected_n_unvoiced: int,
        expected_step_size: int,
) -> np.ndarray:
    """
    Deserialize a *non-compressed* ESPERAudio byte blob produced by:
      Serialize(EsperAudio audio) in LibESPER-V2.Serialization

    Layout (little-endian):
      uint32  fileStandard
      bool    isCompressed
      uint16  nVoiced
      uint16  nUnvoiced
      int32   stepSize
      int32   length
      float32[length * frameSize]  row-major frame data

    Returns
    -------
    np.ndarray
        Shape: (length, expected_frame_size), dtype float32 (little-endian -> native).
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("blob must be bytes-like.")

    mv = memoryview(data)
    pos = 0

    def need(n: int) -> None:
        nonlocal pos
        if pos + n > len(mv):
            raise ValueError(f"Truncated data: need {n} bytes at offset {pos}, have {len(mv) - pos}.")

    need(4)
    file_standard = int(np.frombuffer(mv[pos:pos + 4], dtype="<u4", count=1)[0])
    pos += 4
    if file_standard != expected_file_standard:
        raise ValueError(f"Unexpected fileStandard: got {file_standard}, expected {expected_file_standard}.")

    need(1)
    is_compressed = bool(np.frombuffer(mv[pos:pos + 1], dtype=np.uint8, count=1)[0])
    pos += 1
    if is_compressed:
        raise ValueError("Data is marked as compressed, but this function only supports non-compressed blobs.")

    need(2)
    n_voiced = int(np.frombuffer(mv[pos:pos + 2], dtype="<u2", count=1)[0])
    pos += 2
    if n_voiced != expected_n_voiced:
        raise ValueError(f"Unexpected nVoiced: got {n_voiced}, expected {expected_n_voiced}.")

    need(2)
    n_unvoiced = int(np.frombuffer(mv[pos:pos + 2], dtype="<u2", count=1)[0])
    pos += 2
    if n_unvoiced != expected_n_unvoiced:
        raise ValueError(f"Unexpected nUnvoiced: got {n_unvoiced}, expected {expected_n_unvoiced}.")

    need(4)
    step_size = int(np.frombuffer(mv[pos:pos + 4], dtype="<i4", count=1)[0])
    pos += 4
    if step_size != expected_step_size:
        raise ValueError(f"Unexpected stepSize: got {step_size}, expected {expected_step_size}.")

    need(4)
    length = int(np.frombuffer(mv[pos:pos + 4], dtype="<i4", count=1)[0])
    pos += 4
    if length < 0:
        raise ValueError(f"Invalid length: {length}.")

    expected_frame_size = 1 + 2 * n_voiced + n_unvoiced

    floats_count = length * expected_frame_size
    data_bytes = floats_count * 4
    need(data_bytes)

    frames = np.frombuffer(mv[pos:pos + data_bytes], dtype="<f4", count=floats_count)
    pos += data_bytes

    if pos != len(mv):
        raise ValueError(f"Extra trailing bytes: {len(mv) - pos} bytes remain after reading frames.")

    return frames.reshape((length, expected_frame_size)).astype(np.float32, copy=False)

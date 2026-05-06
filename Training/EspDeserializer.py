import numpy as np


def _read_compressed_blob(
        mv: memoryview,
        start_pos: int,
        expected_file_standard: int,
        expected_n_voiced: int,
        expected_n_unvoiced: int,
        expected_step_size: int,
        expected_temp_comp: int,
        expected_spec_comp: int,
) -> tuple[np.ndarray, int]:
    pos = start_pos

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
    if not is_compressed:
        raise ValueError("Data is marked as uncompressed, but this function only supports compressed blobs.")

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
    temp_comp = int(np.frombuffer(mv[pos:pos + 4], dtype="<i4", count=1)[0])
    pos += 4
    if temp_comp != expected_temp_comp:
        raise ValueError(f"Unexpected temporal compression: got {temp_comp}, expected {expected_temp_comp}.")

    need(4)
    spec_comp = int(np.frombuffer(mv[pos:pos + 4], dtype="<i4", count=1)[0])
    pos += 4
    if spec_comp != expected_spec_comp:
        raise ValueError(f"Unexpected temporal compression: got {spec_comp}, expected {expected_spec_comp}.")

    need(4)
    length = int(np.frombuffer(mv[pos:pos + 4], dtype="<i4", count=1)[0])
    pos += 4
    if length < 0:
        raise ValueError(f"Invalid length: {length}.")

    need(4)
    compressed_length = int(np.frombuffer(mv[pos:pos + 4], dtype="<i4", count=1)[0])
    pos += 4
    expected = int(float(length) / temp_comp)
    if compressed_length != expected:
        raise ValueError(f"Unexpected temporal compression: got {compressed_length}, expected {expected}.")

    expected_frame_size = 1 + n_voiced + int(n_unvoiced / spec_comp)
    floats_count = compressed_length * expected_frame_size
    data_bytes = floats_count * 4
    need(data_bytes)

    frames = np.frombuffer(mv[pos:pos + data_bytes], dtype="<f4", count=floats_count)
    pos += data_bytes
    return frames.reshape((length, expected_frame_size)).astype(np.float32, copy=False), pos


def deserialize_esper_audio_compressed(
        data: bytes,
        expected_file_standard: int,
        expected_n_voiced: int,
        expected_n_unvoiced: int,
        expected_step_size: int,
        expected_temp_comp: int,
        expected_spec_comp: int,
) -> np.ndarray:
    """
    Deserialize a *compressed* ESPERAudio byte blob produced by:
      Serialize(EsperAudio audio) in LibESPER-V2.Serialization

    Layout (little-endian):
      uint32  fileStandard
      bool    isCompressed
      uint16  nVoiced
      uint16  nUnvoiced
      int32   stepSize
      int32   temporalCompression
      int32   spectralCompression
      int32   length
      int32   compressedLength
      float32[compressedLength * frameSize]  row-major frame data

    Returns
    -------
    np.ndarray
        Shape: (length, expected_frame_size), dtype float32 (little-endian -> native).
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("blob must be bytes-like.")

    array, end_pos = _read_compressed_blob(
        memoryview(data),
        0,
        expected_file_standard,
        expected_n_voiced,
        expected_n_unvoiced,
        expected_step_size,
        expected_temp_comp,
        expected_spec_comp
    )
    if end_pos != len(data):
        raise ValueError(f"Extra trailing bytes: {len(data) - end_pos} bytes remain after reading frames.")
    return array


def deserialize_esper_audio_compressed_many(
        data: bytes,
        expected_file_standard: int,
        expected_n_voiced: int,
        expected_n_unvoiced: int,
        expected_step_size: int,
        expected_temp_comp: int,
        expected_spec_comp: int,
) -> list[np.ndarray]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("blob must be bytes-like.")

    mv = memoryview(data)
    arrays = []
    pos = 0
    while pos < len(mv):
        array, pos = _read_compressed_blob(
            mv,
            pos,
            expected_file_standard,
            expected_n_voiced,
            expected_n_unvoiced,
            expected_step_size,
            expected_temp_comp,
            expected_spec_comp
        )
        arrays.append(array)
    if not arrays:
        raise ValueError("No compressed ESPER audio blobs found.")
    return arrays

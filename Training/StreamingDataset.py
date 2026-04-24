import threading
from typing import Optional, Union
import torch
from torch.utils.data import IterableDataset
import zmq

from Training.EspDeserializer import deserialize_esper_audio_compressed


class EsperServerDataset(IterableDataset):
    """
    Torch Dataset that fetches precomputed training samples from the C# ZeroMQ server.

    Protocol (REQ/REP):
      - Must send:  "cfg <NVoiced> <NUnvoiced> <StepSize> <Smoothing> <ExpectedPitch>" once
      - Length:     send "length" -> server replies with files.Length as string
      - Sample:     send any other string -> server replies with a single frame (bytes) sample
    """

    def __init__(
            self,
            n_voiced: int = 33,
            n_unvoiced: int = 257,
            step_size: int = 256,
            temp_comp: int = 1,
            spec_comp: int = 4,
            smoothing: float|str = 0.1,
            expected_pitch: float|str = "null",
            address: str = "tcp://localhost:5555",
            timeout_ms: int = 30_000,
            length_cache: bool = True,
    ):
        super().__init__()

        # Values expected by server Config:
        self.n_voiced = int(n_voiced)
        self.n_unvoiced = int(n_unvoiced)
        self.step_size = int(step_size)
        self.temp_comp = int(temp_comp)
        self.spec_comp = int(spec_comp)
        self.smoothing = smoothing
        self.expected_pitch = expected_pitch

        self.address = str(address)
        self.timeout_ms = int(timeout_ms)
        self.length_cache = bool(length_cache)

        # Lazily created per-process resources (important for DataLoader workers).
        self._ctx: Optional["zmq.Context"] = None
        self._sock: Optional["zmq.Socket"] = None
        self._configured = False
        self._len_cache: Optional[int] = None

        # Guard against accidental multi-threaded access to a single REQ socket.
        self._lock = threading.Lock()

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)

        # Avoid hanging forever.
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)

        self._sock.connect(self.address)

    def _validate_and_parse(self, data: bytes) -> torch.Tensor:
        array = deserialize_esper_audio_compressed(
            data,
            12,
            self.n_voiced,
            self.n_unvoiced,
            self.step_size,
            self.temp_comp,
            self.spec_comp
        )
        tensor = torch.from_numpy(array)
        if tensor.shape[0] > 4096:
            print("WARNING: sample over max context size was truncated.")
            tensor = tensor[:4096]
        return tensor

    def _send_and_recv(self, msg: str, is_meta: bool) -> Union[str, torch.Tensor]:
        """
        Send one REQ and receive one REP. If the server replies with a textual ERROR,
        raise a RuntimeError.
        """
        assert self._sock is not None

        if isinstance(msg, str):
            self._sock.send_string(msg)
        else:
            self._sock.send(msg)

        if is_meta:
            text = self._sock.recv_string()
            if text.startswith("ERROR:"):
                raise RuntimeError(text)
            return text

        data_parts = self._sock.recv_multipart()
        data = b"".join(data_parts)
        # Server can reply with error text as bytes; detect and raise nicely.
        if data.startswith(b"ERROR:"):
            raise RuntimeError(data.decode("utf-8", errors="replace"))
        return self._validate_and_parse(data)

    def _ensure_configured(self) -> None:
        if self._configured:
            return
        self._ensure_connected()

        cfg = f"cfg {self.n_voiced} {self.n_unvoiced} {self.step_size} {self.temp_comp} {self.spec_comp} {self.smoothing} {self.expected_pitch}"
        reply = self._send_and_recv(cfg, is_meta=True)
        if reply != "config received":
            raise RuntimeError(f"Unexpected server reply to cfg: {reply!r}")

        self._configured = True

    def __len__(self) -> int:
        with self._lock:
            if self.length_cache and self._len_cache is not None:
                return self._len_cache

            self._ensure_connected()
            # Length does not require config on the server, but it’s fine either way.
            reply = self._send_and_recv("length", is_meta=True)
            n = int(reply)

            if self.length_cache:
                self._len_cache = n
            return n

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        with self._lock:
            self._ensure_configured()
            return self._send_and_recv("", is_meta=False)

    def close(self) -> None:
        """
        Close the underlying socket. Does NOT send 'exit' (that would stop the shared server).
        """
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                finally:
                    self._sock = None
            self._configured = False

    def __getstate__(self):
        """
        Make this object safe to pickle for multi-worker DataLoader:
        do not pickle live sockets/contexts/locks.
        """
        d = dict(self.__dict__)
        d["_ctx"] = None
        d["_sock"] = None
        d["_configured"] = False
        d["_lock"] = threading.Lock()
        return d
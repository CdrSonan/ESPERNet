using System.Threading.Channels;
using libESPER_V2.Core;
using libESPER_V2.Transforms;
using MathNet.Numerics.LinearAlgebra;
using NAudio.Wave;
using NetMQ;
using NetMQ.Sockets;

var basePath = Environment.GetCommandLineArgs()[1];
var files = Directory.GetFiles(basePath, "*.mp3");
if (files.Length == 0)
{
    Console.Error.WriteLine($"No .mp3 files found in: {basePath}");
}

Config? config = null;

// Bounded buffer (will be capped to <= files.Length to guarantee “no reuse until all used once”)
const int maxBufferedSamples = 32;

// Background pipeline state (recreated when config changes)
SampleBuffer? sampleBuffer = null;

ServerLoop();
return;

void ServerLoop(string address = "tcp://localhost:5555")
{
    using var responder = new ResponseSocket();
    responder.Bind(address);

    while (true)
    {
        var str = responder.ReceiveFrameString();

        if (str == "exit")
        {
            sampleBuffer?.Dispose();
            responder.SendFrame("exit received");
            break;
        }

        if (str.StartsWith("cfg", StringComparison.Ordinal))
        {
            var args = str.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            config = new Config(args);

            // Restart background producers for the new config.
            sampleBuffer?.Dispose();
            sampleBuffer = files.Length == 0
                ? null
                : new SampleBuffer(files, config, maxBufferedSamples);

            responder.SendFrame("config received");
            continue;
        }

        if (str == "length")
        {
            responder.SendFrame(files.Length.ToString());
            continue;
        }

        if (config == null)
        {
            responder.SendFrame("ERROR: received sample request before config");
            continue;
        }

        if (files.Length == 0)
        {
            responder.SendFrame("ERROR: no .mp3 files available");
            continue;
        }

        sampleBuffer ??= new SampleBuffer(files, config, maxBufferedSamples);

        // Block until a precomputed sample is available (generated on background threads).
        var sample = sampleBuffer.GetNextSampleBlocking();
        responder.SendFrame(sample);
    }
}

/// <summary>
/// Asynchronously precomputes samples (expensive EsperTransforms.Forward) on multiple threads and buffers them.
/// Guarantees files are not reused until all files have been used once by:
///  - generating a shuffled permutation of all files per "cycle"
///  - ensuring buffer capacity never exceeds files.Length
/// </summary>
internal sealed class SampleBuffer : IDisposable
{
    private readonly string[] _files;
    private readonly Config _config;

    private readonly CancellationTokenSource _cts = new();
    private readonly Channel<byte[]> _channel;

    private readonly Task[] _workers;
    private readonly SemaphoreSlim _workAvailable = new(0);

    private readonly object _lock = new();

    private int[] _order = [];
    private int _orderPos;

    public SampleBuffer(string[] files, Config config, int requestedCapacity)
    {
        _files = files;
        _config = config;

        // To ensure files are only reused once no more unused files are available, even with buffering,
        // we cap the buffer so it cannot contain items from the next cycle.
        var capacity = Math.Max(1, Math.Min(requestedCapacity, _files.Length));

        _channel = Channel.CreateBounded<byte[]>(new BoundedChannelOptions(capacity)
        {
            SingleReader = false,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.Wait
        });

        ShuffleNewCycle();

        var workerCount = Math.Max(1, Environment.ProcessorCount);
        _workers = new Task[workerCount];

        // Prime the queue with exactly 'capacity' work items.
        _workAvailable.Release(capacity);

        for (var i = 0; i < _workers.Length; i++)
        {
            _workers[i] = Task.Run(WorkerLoop);
        }
    }

    public byte[] GetNextSampleBlocking()
    {
        // As we consume an item, allow producers to create one more (keeping buffer topped up).
        var sample = _channel.Reader.ReadAsync(_cts.Token).AsTask().GetAwaiter().GetResult();
        _workAvailable.Release(1);
        return sample;
    }

    private static byte[] GetSampleFromFile(string filename, Config config)
    {
        using var reader = new Mp3FileReader(filename);
        var audio = reader.ToSampleProvider();
        var sampleCount = (int)(reader.Length / (reader.WaveFormat.BitsPerSample / 8));
        var waveform = new float[sampleCount];
        audio.Read(waveform, 0, sampleCount);

        var sampleConfig = new EsperAudioConfig((ushort)config.NVoiced, (ushort)config.NUnvoiced, config.StepSize);
        var forwardConfig = new EsperForwardConfig(config.Smoothing, Vector<float>.Build.Dense(1, config.ExpectedPitch));

        var esperAudio = EsperTransforms.Forward(
            Vector<float>.Build.DenseOfArray(waveform),
            sampleConfig,
            forwardConfig);

        return Serialization.Serialize(esperAudio);
    }

    private async Task WorkerLoop()
    {
        try
        {
            while (!_cts.IsCancellationRequested)
            {
                await _workAvailable.WaitAsync(_cts.Token).ConfigureAwait(false);

                var fileIndex = GetNextFileIndex();
                var filename = _files[fileIndex];

                // Expensive CPU work done off-thread here.
                var sample = GetSampleFromFile(filename, _config);

                await _channel.Writer.WriteAsync(sample, _cts.Token).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
            // normal on Dispose
        }
        catch (Exception ex)
        {
            // If a worker dies, log what happened, then complete the channel so the server doesn't hang forever.
            await Console.Error.WriteAsync($"Worker encountered exception: {ex}");
            _channel.Writer.TryComplete();
        }
    }

    private int GetNextFileIndex()
    {
        lock (_lock)
        {
            if (_orderPos >= _order.Length)
            {
                ShuffleNewCycle();
            }

            return _order[_orderPos++];
        }
    }

    private void ShuffleNewCycle()
    {
        _order = Enumerable.Range(0, _files.Length).ToArray();

        // Fisher–Yates shuffle (non-deterministic between runs)
        for (var i = _order.Length - 1; i > 0; i--)
        {
            var j = Random.Shared.Next(i + 1);
            (_order[i], _order[j]) = (_order[j], _order[i]);
        }

        _orderPos = 0;
    }

    public void Dispose()
    {
        _cts.Cancel();
        _channel.Writer.TryComplete();

        try
        {
            Task.WaitAll(_workers, TimeSpan.FromSeconds(2));
        }
        catch
        {
            // ignore on shutdown
        }

        _cts.Dispose();
        _workAvailable.Dispose();
    }
}

public class Config
{
    public readonly int NVoiced = 10;
    public readonly int NUnvoiced = 10;
    public readonly int StepSize = 10;
    public readonly float Smoothing = 0.5f;
    public readonly float ExpectedPitch = 440;

    public Config(string[] args)
    {
        NVoiced = int.Parse(args[1]);
        NUnvoiced = int.Parse(args[2]);
        StepSize = int.Parse(args[3]);
        Smoothing = float.Parse(args[4]);
        ExpectedPitch = float.Parse(args[5]);
    }
}

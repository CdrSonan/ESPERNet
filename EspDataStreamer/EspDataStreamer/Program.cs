using System.Threading.Channels;
using libESPER_V2.Core;
using libESPER_V2.Effects;
using libESPER_V2.Transforms;
using MathNet.Numerics.Interpolation;
using MathNet.Numerics.LinearAlgebra;
using NAudio.Wave;
using NetMQ;
using NetMQ.Sockets;
using System.Reflection;
using EspDataStreamer;

var basePath = Environment.GetCommandLineArgs()[1];
var bindAddress = Environment.GetCommandLineArgs()[2];
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

ServerLoop(bindAddress);
return;

void ServerLoop(string address = "tcp://localhost:5555")
{
    using var responder = new ResponseSocket();
    responder.Bind(address);

    while (true)
    {
        var str = responder.ReceiveFrameString();
        Console.WriteLine($"Received ZeroMQ Message: {str}");

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

        const int chunkSize = 65536; // 64 KiB
        var chunkCount = (sample.Length + chunkSize - 1) / chunkSize;
        var sampleParts = new List<byte[]>(chunkCount);

        for (var i = 0; i < sample.Length; i += chunkSize)
        {
            var currentChunkSize = Math.Min(chunkSize, sample.Length - i);
            var chunk = new byte[currentChunkSize];
            Array.Copy(sample, i, chunk, 0, currentChunkSize);
            sampleParts.Add(chunk);
        }
        Console.WriteLine($"Sending {sampleParts.Count} chunks");
        responder.SendMultipartBytes(sampleParts.ToArray());
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
    private static readonly MethodInfo[] EffectMethods = typeof(Effects)
        .GetMethods(BindingFlags.Public | BindingFlags.Static)
        .Where(method =>
        {
            if (method.ReturnType != typeof(void))
            {
                return false;
            }

            var parameters = method.GetParameters();
            if (parameters.Length < 2 || parameters[0].ParameterType != typeof(EsperAudio))
            {
                return false;
            }

            return parameters.Skip(1).All(parameter => parameter.ParameterType == typeof(Vector<float>));
        })
        .ToArray();

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

    private static EsperAudio CreateAugmentedAudio(EsperAudio sourceAudio)
    {
        return Augmentation.Augment(sourceAudio, 2);
    }

    private static byte[] GetSampleFromFile(string filename, Config config)
    {
        Console.WriteLine($"Reading {filename}");
        using var reader = new Mp3FileReader(filename);
        var audio = reader.ToSampleProvider();
        var sampleRate = reader.WaveFormat.SampleRate;
        var sampleCount = (int)(reader.Length / (reader.WaveFormat.BitsPerSample / 8));
        var waveform = new float[sampleCount];
        audio.Read(waveform, 0, sampleCount);
        var x = Vector<double>.Build.Dense(sampleCount, i => (double)i / sampleRate).ToArray();
        var y = Vector<double>.Build.Dense(sampleCount, i => waveform[i]).ToArray();
        var interpolator = CubicSpline.InterpolatePchip(x, y);
        var resampFactor = 48000d / sampleRate;
        var resampled = Vector<double>.Build.Dense((int)(sampleCount * resampFactor), i => interpolator.Interpolate(i / 48000d));

        var sampleConfig = new EsperAudioConfig((ushort)config.NVoiced, (ushort)config.NUnvoiced, config.StepSize);
        var forwardConfig = new EsperForwardConfig
        {
            PitchOscillatorDamping = config.Smoothing,
            ExpectedPitch = config.ExpectedPitch == null ? null : Vector<float>.Build.Dense(1, config.ExpectedPitch.Value)
        };

        var esperAudio = EsperTransforms.Forward(
            Vector<float>.Build.Dense(resampled.Count, i => (float)resampled[i]),
            sampleConfig,
            forwardConfig);
        var compressedSamples = new CompressedEsperAudio[config.NAugs + 1];
        compressedSamples[0] = Compression.Compress(esperAudio, config.TempComp, config.SpecComp, 1e-5f);
        for (var i = 1; i < compressedSamples.Length; i++)
        {
            var augmented = CreateAugmentedAudio(esperAudio);
            compressedSamples[i] = Compression.Compress(augmented, config.TempComp, config.SpecComp, 1e-5f);
        }

        var serializedSamples = new byte[compressedSamples.Length][];
        var totalLength = 0;
        for (var i = 0; i < compressedSamples.Length; i++)
        {
            serializedSamples[i] = Serialization.Serialize(compressedSamples[i]);
            totalLength += serializedSamples[i].Length;
        }

        var payload = new byte[totalLength];
        var offset = 0;
        foreach (var serializedSample in serializedSamples)
        {
            Buffer.BlockCopy(serializedSample, 0, payload, offset, serializedSample.Length);
            offset += serializedSample.Length;
        }

        Console.WriteLine(
            $"Read {filename} ({compressedSamples[0].Length} frames, {compressedSamples.Length} variants)");
        return payload;
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

                try
                {
                    // Expensive CPU work done off-thread here.
                    var sample = GetSampleFromFile(filename, _config);

                    await _channel.Writer.WriteAsync(sample, _cts.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    await Console.Error.WriteLineAsync($"Failed to process '{filename}': {ex.Message}");

                    try
                    {
                        var erroredDirectory = Path.Combine(Path.GetDirectoryName(filename) ?? ".", "errored");
                        Directory.CreateDirectory(erroredDirectory);

                        var destinationPath = Path.Combine(erroredDirectory, Path.GetFileName(filename));

                        if (File.Exists(destinationPath))
                        {
                            var fileNameWithoutExtension = Path.GetFileNameWithoutExtension(filename);
                            var extension = Path.GetExtension(filename);
                            destinationPath = Path.Combine(
                                erroredDirectory,
                                $"{fileNameWithoutExtension}_{DateTime.UtcNow:yyyyMMddHHmmssfff}{extension}");
                        }

                        File.Move(filename, destinationPath);
                        await Console.Error.WriteLineAsync($"Moved errored file to '{destinationPath}'.");
                    }
                    catch (Exception moveEx)
                    {
                        await Console.Error.WriteLineAsync($"Failed to move '{filename}' to errored folder: {moveEx.Message}");
                    }

                }
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
    public readonly int NVoiced;
    public readonly int NUnvoiced;
    public readonly int StepSize;
    public readonly int TempComp;
    public readonly int SpecComp;
    public readonly float? Smoothing;
    public readonly float? ExpectedPitch;
    public readonly int NAugs;

    public Config(string[] args)
    {
        NVoiced = int.Parse(args[1]);
        NUnvoiced = int.Parse(args[2]);
        StepSize = int.Parse(args[3]);
        TempComp = int.Parse(args[4]);
        SpecComp = int.Parse(args[5]);
        Smoothing = args[6] == "null" ? null : float.Parse(args[6]);
        ExpectedPitch = args[7] == "null" ? null : float.Parse(args[7]);
        NAugs = int.Parse(args[8]);
        if (NAugs < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(NAugs), "NAugs must be >= 0.");
        }
    }
}

using System;
using System.Threading;
using libESPER_V2.Core;
using libESPER_V2.Transforms;
using MathNet.Numerics.LinearAlgebra;
using NAudio.Wave;
using NetMQ;
using NetMQ.Sockets;

var basePath = Environment.GetCommandLineArgs()[1];
var files = Directory.GetFiles(basePath, "*.wav");
Config? config = null;
ServerLoop();
return;

void ServerLoop(string address="tcp://localhost:5555")
{
    using var responder = new ResponseSocket();
    responder.Bind(address);
    while (true)
    {
        var str = responder.ReceiveFrameString();
        if (str == "exit")
        {
            responder.SendFrame("exit received");
            break;
        }
        if (str.StartsWith("cfg"))
        {
            var args = str.Split(' ');
            config = new Config(args);
            responder.SendFrame("config received");
            continue;
        }

        if (config == null)
        {
            responder.SendFrame("ERROR: config not yet received");
            continue;
        }
        var sample = GetSampleFromFile("", config);
        responder.SendFrame(sample);
    }
}

byte[] GetSampleFromFile(string filename, Config config)
{
    using var reader = new WaveFileReader(filename);
    var waveform = new float[reader.SampleCount];

    for (var i = 0; i < waveform.Length; i++)
    {
        var frame = reader.ReadNextSampleFrame();
        waveform[i] = frame[0];
    }

    var sampleConfig = new EsperAudioConfig((ushort)config.NVoiced, (ushort)config.NUnvoiced, config.StepSize);
    var forwardConfig = new EsperForwardConfig(config.Smoothing, Vector<float>.Build.Dense(1, config.ExpectedPitch));

    var esperAudio = EsperTransforms.Forward(
        Vector<float>.Build.DenseOfArray(waveform),
        sampleConfig,
        forwardConfig);

    return Serialization.Serialize(esperAudio);
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
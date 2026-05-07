using libESPER_V2.Core;
using libESPER_V2.Effects;
using MathNet.Numerics.LinearAlgebra;

namespace EspDataStreamer;

public static class Augmentation
{
    private static readonly (Action<EsperAudio, Vector<float>>, float, float)[] Fx = [
    (Effects.Breathiness, -1, 0.8f),
    (Effects.Brightness, -1, 1),
    (Effects.Dynamics, -1, 1),
    (Effects.FormantShift, -1, 1),
    (Effects.Growl, 0, 1),
    (Effects.Mouth, -1, 1),
    (Effects.Roughness, -1, 1),
    (Effects.Steadiness, -1, 1)
    ];

    private static void ApplyRandomPitchShift(EsperAudio audio)
    {
        var shift = (float)Math.Pow(2, Random.Shared.NextDouble() * 2 - 1);
        var sourcePitch = audio.GetPitch();
        var targetPitch = sourcePitch * shift;
        Effects.PitchShift(audio, targetPitch);
    }

    private static void ApplyRandomFx(EsperAudio audio)
    {
        var (effect, minStrength, maxStrength) = Fx[Random.Shared.Next(Fx.Length)];
        var strength = minStrength + (float)(Random.Shared.NextDouble() * (maxStrength - minStrength));
        var strengthVector = Vector<float>.Build.Dense(audio.Length, strength);
        effect(audio, strengthVector);
    }

    public static EsperAudio Augment(EsperAudio audio, uint nAugs)
    {
        ApplyRandomPitchShift(audio);
        for (var i = 0; i < nAugs; i++)
        {
            ApplyRandomFx(audio);
        }
        return audio;
    }
}

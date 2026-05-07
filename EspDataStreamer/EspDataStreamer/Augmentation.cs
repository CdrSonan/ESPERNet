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

    private static void ApplyFx(EsperAudio audio, (Action<EsperAudio, Vector<float>> effect, float minStrength, float maxStrength) fx)
    {
        var (effect, minStrength, maxStrength) = fx;
        var strength = minStrength + (float)(Random.Shared.NextDouble() * (maxStrength - minStrength));
        var strengthVector = Vector<float>.Build.Dense(audio.Length, strength);
        effect(audio, strengthVector);
    }

    public static EsperAudio Augment(EsperAudio audio, uint nAugs)
    {
        if (nAugs > Fx.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(nAugs), "nAugs must be <= " + Fx.Length);
        }
        ApplyRandomPitchShift(audio);

        var fxIndices = new int[Fx.Length];
        for (var i = 0; i < Fx.Length; i++)
        {
            fxIndices[i] = i;
        }

        for (var i = 0; i < nAugs; i++)
        {
            var randomIndex = Random.Shared.Next(i, Fx.Length);
            (fxIndices[i], fxIndices[randomIndex]) = (fxIndices[randomIndex], fxIndices[i]);
        }

        for (var i = 0; i < nAugs; i++)
        {
            ApplyFx(audio, Fx[fxIndices[i]]);
        }
        return audio;
    }
}

// ExperimentController.cs  (full corrected file)
using UnityEngine;
using System.Collections;
using System.IO;
using System.Diagnostics;
using System;
using System.Reflection;
using Debug = UnityEngine.Debug;
using System.Linq;

public class ExperimentController : MonoBehaviour
{
    [Header("Tracer / scene")]
    public RayComputeManager tracer; // assign the RayComputeManager component here
    public Texture2D groundTruth;    // high-SPP EXR or PNG reference (import with Read/Write enabled)

    [Header("Experiment settings")]
    public string samplingMethod = "uniform"; // "uniform" or "importance"
    public int[] sppList = new int[] { 1, 4, 16, 64, 128 };
    public string outputFolder = "RenderOutputs";
    public bool saveAsEXR = true; // true -> EXR, false -> PNG
    public bool alsoSavePreviewPNG = true; // save tone-mapped PNG for quick viewing

    private string assetPath;

    void Start()
    {
        if (tracer == null)
        {
            Debug.LogError("ExperimentController: assign the RayComputeManager component 'tracer' in the Inspector.");
            enabled = false;
            return;
        }

        assetPath = Application.dataPath;
        Directory.CreateDirectory(Path.Combine(assetPath, outputFolder));
        StartCoroutine(RunExperimentsCoroutine());
    }

    IEnumerator RunExperimentsCoroutine()
    {
        string csvPath = Path.Combine(assetPath, outputFolder, "results.csv");
        using (var sw = new StreamWriter(csvPath, false))
        {
            sw.WriteLine("spp,render_time_s,mean_variance,mse,psnr_dB,sampling_method,image_file");
        }

        foreach (int spp in sppList)
        {
            Debug.Log($"[ExperimentController] Starting experiment SPP={spp}, method={samplingMethod}");

            // Set sampling mode (if tracer supports it)
            try { tracer.SetSamplingMethod(samplingMethod); } catch { /* no-op */ }

            // enable accumulation (if field exists)
            try { tracer.accumulate = true; } catch { /* no-op */ }

            // Reset accumulation (use wrapper if available)
            try { tracer.ResetAccumulation(); } catch { try { tracer.ResetAccumulatedRender(); } catch { /* ignore */ } }

            // allow one frame for reset
            yield return null;

            var swatch = new Stopwatch();
            swatch.Start();

            // Wait until tracer reports enough frames (numAccumulatedFrames >= spp)
            while (true)
            {
                int frames = -1;
                try { frames = tracer.numAccumulatedFrames; } catch { frames = -1; }

                if (frames >= spp) break;

                if (frames < 0)
                {
                    // fallback wait
                    yield return new WaitForSeconds(0.1f);
                }
                else
                {
                    yield return null;
                }
            }

            swatch.Stop();
            double renderSeconds = swatch.Elapsed.TotalSeconds;

            // Read accumulatedResult or raytraceFrameTex
            Texture2D tex = null;
            try
            {
                RenderTexture rt = tracer.accumulatedResult;
                tex = ReadRenderTextureToTexture2D(rt);
            }
            catch
            {
                try
                {
                    RenderTexture rt2 = tracer.raytraceFrameTex;
                    tex = ReadRenderTextureToTexture2D(rt2);
                }
                catch (System.Exception e)
                {
                    Debug.LogError("ExperimentController: Could not read tracer textures. " + e.Message);
                }
            }

            // Compute MSE/PSNR using robust method
            double mse = -1;
            double psnr = -1;
            if (groundTruth != null && tex != null)
            {
                var result = ComputeMSEandPSNR(tex, groundTruth);
                if (result.ok)
                {
                    mse = result.mse;
                    psnr = result.psnr;
                    Debug.Log($"Computed MSE={mse:E6}, PSNR={psnr:F3} dB for SPP={spp}");
                }
                else
                {
                    Debug.LogWarning("Failed to compute MSE/PSNR for SPP=" + spp);
                }
            }
            else
            {
                Debug.Log("Ground truth not set or tex null — skipping PSNR computation for SPP=" + spp);
            }

            // mean variance: use reflection to detect GetPerPixelVariance() if tracer implements it
            double meanVar = -1;
            try
            {
                MethodInfo mi = tracer.GetType().GetMethod("GetPerPixelVariance", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (mi != null)
                {
                    object resultVar = mi.Invoke(tracer, null);
                    if (resultVar is float[] fa)
                    {
                        double sum = 0;
                        for (int i = 0; i < fa.Length; i++) sum += fa[i];
                        meanVar = sum / fa.Length;
                    }
                    else if (resultVar is double[] da)
                    {
                        double sum = 0;
                        for (int i = 0; i < da.Length; i++) sum += da[i];
                        meanVar = sum / da.Length;
                    }
                    else if (resultVar is System.Collections.IEnumerable ie)
                    {
                        double sum = 0;
                        long count = 0;
                        foreach (object o in ie)
                        {
                            try { sum += Convert.ToDouble(o); count++; } catch { }
                        }
                        if (count > 0) meanVar = sum / count;
                    }
                    else
                    {
                        Debug.LogWarning("GetPerPixelVariance returned unsupported type: " + (resultVar == null ? "null" : resultVar.GetType().ToString()));
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("Error invoking GetPerPixelVariance via reflection: " + e.Message);
            }

            // Save image(s)
            string fname = $"render_{samplingMethod}_spp{spp}.exr";
            string outPath = Path.Combine(assetPath, outputFolder, fname);
            if (tex != null)
            {
                try
                {
                    if (saveAsEXR)
                    {
                        var bytes = tex.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);
                        File.WriteAllBytes(outPath, bytes);
                        Debug.Log("[ExperimentController] Saved EXR -> " + outPath + " (bytes: " + bytes.Length + ")");
                    }
                    else
                    {
                        var png = tex.EncodeToPNG();
                        outPath = outPath.Replace(".exr", ".png");
                        File.WriteAllBytes(outPath, png);
                        Debug.Log("[ExperimentController] Saved PNG -> " + outPath + " (bytes: " + png.Length + ")");
                    }

                    // Also save a tone-mapped preview PNG for quick viewing (optional)
                    if (alsoSavePreviewPNG)
                    {
                        try
                        {
                            string previewPath = Path.Combine(assetPath, outputFolder, $"render_{samplingMethod}_spp{spp}_preview.png");
                            SaveToneMappedPNG(tex, previewPath);
                            Debug.Log("[ExperimentController] Saved tone-mapped preview -> " + previewPath);
                        }
                        catch (Exception ePreview)
                        {
                            Debug.LogWarning("Could not save preview PNG: " + ePreview.Message);
                        }
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError("Failed to save image: " + e.Message);
                }
            }

            // Append CSV row
            try
            {
                using (var sw = new StreamWriter(Path.Combine(assetPath, outputFolder, "results.csv"), true))
                {
                    sw.WriteLine($"{spp},{renderSeconds},{meanVar},{mse},{psnr},{samplingMethod},{fname}");
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Failed to write CSV row: " + e.Message);
            }

            Debug.Log($"Experiment SPP={spp} done: time={renderSeconds:F3}s PSNR={(double.IsPositiveInfinity(psnr) ? 100.0 : psnr):F2} dB var={meanVar}");
            yield return null;
        }

        Debug.Log("All experiments finished. CSV in " + Path.Combine(assetPath, outputFolder));
    }

    // ---------- helper: resample GT to renderer size ----------
    Texture2D ResampleTextureToTexture2D(Texture2D src, int targetW, int targetH)
    {
        if (src == null) return null;
        if (src.width == targetW && src.height == targetH) return src;

        RenderTexture rt = RenderTexture.GetTemporary(targetW, targetH, 0, RenderTextureFormat.ARGBFloat);
        RenderTexture prev = RenderTexture.active;

        // If src is readable, we can use Graphics.Blit after creating a temporary material-less copy
        // Graphics.Blit(Texture, RT) works in recent Unity versions
        Graphics.Blit(src, rt);

        RenderTexture.active = rt;
        Texture2D outTex = new Texture2D(targetW, targetH, TextureFormat.RGBAFloat, false, true);
        outTex.ReadPixels(new Rect(0, 0, targetW, targetH), 0, 0);
        outTex.Apply();

        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);
        return outTex;
    }

    // ---------- compute MSE and PSNR robustly ----------
    (bool ok, double mse, double psnr) ComputeMSEandPSNR(Texture2D img, Texture2D refTex)
    {
        if (img == null || refTex == null)
        {
            Debug.LogWarning("ComputeMSEandPSNR: one of textures is null");
            return (false, -1, -1);
        }

        // If refTex size != img size -> resample ref to img size
        Texture2D gtForCompare = refTex;
        if (refTex.width != img.width || refTex.height != img.height)
        {
            Debug.Log("Ground truth and render size differ. Resampling GT from " + refTex.width + "x" + refTex.height + " to " + img.width + "x" + img.height);
            gtForCompare = ResampleTextureToTexture2D(refTex, img.width, img.height);
            if (gtForCompare == null)
            {
                Debug.LogWarning("Failed to resample GT.");
                return (false, -1, -1);
            }
        }

        try
        {
            Color[] A = img.GetPixels();
            Color[] B = gtForCompare.GetPixels();
            int n = A.Length;
            double mse = 0.0;
            double maxRef = 0.0;
            for (int i = 0; i < n; i++)
            {
                double dr = A[i].r - B[i].r;
                double dg = A[i].g - B[i].g;
                double db = A[i].b - B[i].b;
                mse += (dr * dr + dg * dg + db * db) / 3.0;
                maxRef = Math.Max(maxRef, Math.Max(B[i].r, Math.Max(B[i].g, B[i].b)));
            }
            mse /= n;

            // choose max_pixel for PSNR: use maxRef or 1.0 if maxRef small
            double maxPixel = Math.Max(maxRef, 1.0);
            double psnr = (mse > 0.0) ? (20.0 * Math.Log10(maxPixel) - 10.0 * Math.Log10(mse)) : double.PositiveInfinity;

            return (true, mse, psnr);
        }
        catch (Exception e)
        {
            Debug.LogWarning("ComputeMSEandPSNR failed: " + e.Message);
            return (false, -1, -1);
        }
    }

    // ---------- read RT to Texture2D and normalize by sample count ----------
    Texture2D ReadRenderTextureToTexture2D(RenderTexture rt)
    {
        if (rt == null) return null;

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBAFloat, false, true);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);

        // NORMALIZE THE IMAGE BEFORE SAVING
        int n = 1;
        try { n = Math.Max(1, tracer.numAccumulatedFrames); } catch { }

        Color[] px = tex.GetPixels();
        for (int i = 0; i < px.Length; i++)
        {
            px[i] /= (float)n;   // Convert SUM -> AVERAGE
        }
        tex.SetPixels(px);
        tex.Apply();

        RenderTexture.active = prev;
        return tex;
    }

    // ---------- small helper: save tone-mapped PNG ----------
    void SaveToneMappedPNG(Texture2D hdrTex, string outPath)
    {
        if (hdrTex == null) return;
        Color[] px = hdrTex.GetPixels();
        for (int i = 0; i < px.Length; i++)
        {
            Color c = px[i];
            // Reinhard tone mapping (per-channel) then gamma to sRGB
            c.r = c.r / (1.0f + c.r);
            c.g = c.g / (1.0f + c.g);
            c.b = c.b / (1.0f + c.b);
            c.r = Mathf.Pow(Mathf.Clamp01(c.r), 1.0f / 2.2f);
            c.g = Mathf.Pow(Mathf.Clamp01(c.g), 1.0f / 2.2f);
            c.b = Mathf.Pow(Mathf.Clamp01(c.b), 1.0f / 2.2f);
            px[i] = c;
        }
        Texture2D ldr = new Texture2D(hdrTex.width, hdrTex.height, TextureFormat.RGBA32, false);
        ldr.SetPixels(px); ldr.Apply();
        File.WriteAllBytes(outPath, ldr.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(ldr);
    }
}

using UnityEngine;
using System.Collections.Generic;
using Seb.AccelerationStructures;
using Seb.Helpers;
using UnityEngine.Experimental.Rendering;

public class RayComputeManager : MonoBehaviour
{
    [Header("Main Settings")]
    public bool rayTracingEnabled = true;

    public bool accumulate = true;
    public BVH.Quality bvhQuality = BVH.Quality.High;

    [SerializeField, Range(0, 32)] int maxBounceCount = 4;
    [SerializeField] int numRaysPerPixel = 1;
    [SerializeField, Min(0)] float defocusStrength = 0;
    [SerializeField, Min(0)] float divergeStrength = 0.3f;
    [Min(0)] public float focusDistance = 1;

    [Header("Sky Settings")]
    public bool useSky;

    [SerializeField] float sunFocus = 500;
    [SerializeField] float sunIntensity = 10;
    [SerializeField] Color sunColor = Color.white;
    public Transform sunTransform;

    [Header("Debug Settings")]
    public string screenshotName = "rayScreenshot";
    public Vector4 debugParams;

    [Header("References")]
    [SerializeField] ComputeShader rayComputeShader;

    [Header("Info")]
    public int numAccumulatedFrames;

    public int renderSeed;

    [SerializeField] float info_timeSinceReset;
    [SerializeField] Vector2Int screenSize;

    // Buffers
    ComputeBuffer triangleBuffer;
    ComputeBuffer nodeBuffer;
    ComputeBuffer modelBuffer;

    MeshInfo[] meshInfo;
    Model[] models;
    bool hasBVH;

    [HideInInspector] public RenderTexture raytraceFrameTex;
    [HideInInspector] public RenderTexture accumulatedResult;
    public bool IsRendering => Application.isPlaying && rayTracingEnabled;

    const int kernelRayTrace = 0;
    const int kernelResetAccumulated = 1;

    // ---------- NEW: CPU variance accumulation fields ----------
    [Header("Variance / Diagnostics")]
    [Tooltip("If true, the tracer will read back each frame (slow) and compute per-pixel luminance variance on the CPU.")]
    public bool computeVarianceCPU = true;

    // Temporary readback texture reused each frame
    Texture2D _tmpReadbackTex = null;

    // Running sums (S1) and sums-of-squares (S2) for luminance per pixel
    float[] _sumLuma = null;
    float[] _sumSqLuma = null;
    int _varianceTexW = 0;
    int _varianceTexH = 0;
    // ---------------------------------------------------------

    private void OnEnable()
    {
        hasBVH = false;
        renderSeed = new System.Random().Next();

        ResetAccumulatedRender();
    }

    public void ResetAccumulatedRender()
    {
        // Reset counters and buffers
        numAccumulatedFrames = 0;
        info_timeSinceReset = 0;
        InitFrame();

        // zero GPU accumulation via compute kernel (if available)
        try
        {
            Seb.Helpers.ComputeHelper.Dispatch(rayComputeShader, accumulatedResult.width, accumulatedResult.height, kernelIndex: kernelResetAccumulated);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("ResetAccumulatedRender: dispatch failed: " + e.Message);
            // fallback clear CPU-side if needed below
        }

        // Zero CPU-side variance accumulators if in use
        if (computeVarianceCPU && _sumLuma != null && _sumSqLuma != null)
        {
            int cnt = _sumLuma.Length;
            for (int i = 0; i < cnt; ++i) { _sumLuma[i] = 0f; _sumSqLuma[i] = 0f; }
        }
    }

    // ---------- BEGIN: tiny public helper wrappers for ExperimentController ----------

    // Public sampling mode (0 = uniform, 1 = importance). ExperimentController will call SetSamplingMethod.
    [HideInInspector] public int samplingMode = 0;

    // ResetAccumulation: clears the accumulated render and resets the frame/sample count to 0.
    // ExperimentController calls this to start a fresh accumulation for a given SPP.
    public void ResetAccumulation()
    {
        // Reset counter to 0 (ExperimentController expects to wait until numAccumulatedFrames >= spp)
        numAccumulatedFrames = 0;
        info_timeSinceReset = 0;
        // Ensure textures and buffers are initialized
        InitFrame();

        // Call the existing compute shader kernel to clear the accumulated result (same behavior as ResetAccumulatedRender)
        try
        {
            Seb.Helpers.ComputeHelper.Dispatch(rayComputeShader, accumulatedResult.width, accumulatedResult.height, kernelIndex: kernelResetAccumulated);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("ResetAccumulation: failed to dispatch reset kernel: " + e.Message);
            // fallback: clear on CPU (cheap for debugging, but slower)
            if (accumulatedResult != null)
            {
                RenderTexture prev = RenderTexture.active;
                RenderTexture.active = accumulatedResult;
                GL.Clear(true, true, Color.black);
                RenderTexture.active = prev;
            }
        }

        // Zero CPU accumulators too
        if (computeVarianceCPU && _sumLuma != null && _sumSqLuma != null)
        {
            int cnt = _sumLuma.Length;
            for (int i = 0; i < cnt; ++i) { _sumLuma[i] = 0f; _sumSqLuma[i] = 0f; }
        }
    }

    // SetSamplingMethod: simple name -> integer mapper. Also pushes the value to the compute shader.
    public void SetSamplingMethod(string name)
    {
        if (string.IsNullOrEmpty(name)) return;

        name = name.ToLowerInvariant();
        if (name == "uniform") samplingMode = 0;
        else if (name == "importance") samplingMode = 1;
        else
        {
            // default fallback
            samplingMode = 0;
            Debug.Log("SetSamplingMethod: unknown name '" + name + "' -> defaulting to uniform (0).");
        }

        // push to shader (safe even if shader doesn't read the variable yet)
        try
        {
            rayComputeShader.SetInt("SamplingMode", samplingMode);
        }
        catch (System.Exception)
        {
            // not fatal — shader may not declare SamplingMode yet
        }
    }

    // (Optional) convenience getter that ExperimentController might use; your controller already uses tracer.numAccumulatedFrames directly
    public int GetNumAccumulatedFrames()
    {
        return numAccumulatedFrames;
    }

    // ---------- END: tiny public helper wrappers ----------

    private void Update()
    {
        RenderFrame();
        HandleInput();
    }

    void RenderFrame()
    {
        if (!IsRendering) return;

        InitFrame();

        Seb.Helpers.ComputeHelper.Dispatch(rayComputeShader, raytraceFrameTex.width, raytraceFrameTex.height, kernelIndex: kernelRayTrace);

        info_timeSinceReset += Time.deltaTime;

        // ---------- NEW: CPU-side readback for variance accumulation (slow) ----------
        if (computeVarianceCPU)
        {
            // read raytraceFrameTex (single sample for this frame)
            if (raytraceFrameTex != null)
            {
                RenderTexture prev = RenderTexture.active;
                RenderTexture.active = raytraceFrameTex;

                if (_tmpReadbackTex == null || _tmpReadbackTex.width != raytraceFrameTex.width || _tmpReadbackTex.height != raytraceFrameTex.height)
                {
                    if (_tmpReadbackTex != null) UnityEngine.Object.DestroyImmediate(_tmpReadbackTex);
                    _tmpReadbackTex = new Texture2D(raytraceFrameTex.width, raytraceFrameTex.height, TextureFormat.RGBAFloat, false, true);
                }

                _tmpReadbackTex.ReadPixels(new Rect(0, 0, raytraceFrameTex.width, raytraceFrameTex.height), 0, 0);
                _tmpReadbackTex.Apply();

                RenderTexture.active = prev;

                Color[] px = _tmpReadbackTex.GetPixels();
                int count = px.Length;
                if (_sumLuma == null || _sumSqLuma == null || _sumLuma.Length != count)
                {
                    _sumLuma = new float[count];
                    _sumSqLuma = new float[count];
                    for (int i = 0; i < count; ++i) { _sumLuma[i] = 0f; _sumSqLuma[i] = 0f; }
                }

                // accumulate per-pixel luminance (linear)
                for (int i = 0; i < count; ++i)
                {
                    float r = px[i].r;
                    float g = px[i].g;
                    float b = px[i].b;
                    float l = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                    _sumLuma[i] += l;
                    _sumSqLuma[i] += l * l;
                }
            }
        }
        // ---------- end CPU accumulation ----------

        if (accumulate) numAccumulatedFrames++;
    }


    void HandleInput()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            ResetAccumulatedRender();
            Debug.Log("Reset render");
        }

        if (Input.GetKeyDown(KeyCode.S))
        {
            string path = System.IO.Path.Combine(Application.persistentDataPath, screenshotName + ".png");
            ScreenCapture.CaptureScreenshot(path);
            Debug.Log("Screenshot: " + path);
        }
    }


    void InitFrame()
    {
        InitTexturesAndBuffers();
        models = FindObjectsByType<Model>(FindObjectsInactive.Exclude, FindObjectsSortMode.InstanceID);

        InitBVH();
        UpdateModels();
        UpdateCameraParams(Camera.main);
        SetShaderParams();
    }

    void InitTexturesAndBuffers()
    {
        int width = Screen.width;
        int height = Screen.height;
        screenSize = new Vector2Int(width, height);

        Seb.Helpers.ComputeHelper.CreateRenderTexture(ref raytraceFrameTex, width, height, FilterMode.Bilinear, GraphicsFormat.R32G32B32A32_SFloat, "Raytrace Frame");
        Seb.Helpers.ComputeHelper.CreateRenderTexture(ref accumulatedResult, width, height, FilterMode.Bilinear, GraphicsFormat.R32G32B32A32_SFloat, "Raytrace Accumulated");

        rayComputeShader.SetTexture(kernelRayTrace, "FrameRender", raytraceFrameTex);
        rayComputeShader.SetTexture(kernelRayTrace, "AccumulatedRender", accumulatedResult);
        rayComputeShader.SetTexture(kernelResetAccumulated, "AccumulatedRender", accumulatedResult);

        rayComputeShader.SetInts("Resolution", raytraceFrameTex.width, raytraceFrameTex.height);
        rayComputeShader.SetVector("debugParams", debugParams);

        // ---------- NEW: prepare CPU variance buffers (if enabled) ----------
        if (computeVarianceCPU)
        {
            int w = raytraceFrameTex.width;
            int h = raytraceFrameTex.height;
            if (_varianceTexW != w || _varianceTexH != h || _sumLuma == null)
            {
                _varianceTexW = w; _varianceTexH = h;
                int count = w * h;
                _sumLuma = new float[count];
                _sumSqLuma = new float[count];
                if (_tmpReadbackTex != null) UnityEngine.Object.DestroyImmediate(_tmpReadbackTex);
                _tmpReadbackTex = new Texture2D(w, h, TextureFormat.RGBAFloat, false, true);
                for (int i = 0; i < count; ++i) { _sumLuma[i] = 0f; _sumSqLuma[i] = 0f; }
                // ensure we start counting from zero
                numAccumulatedFrames = 0;
            }
        }
        // ---------------------------------------------------------------
    }

    void InitBVH()
    {
        if (hasBVH) return;

        hasBVH = true;
        var data = CreateAllMeshData(models);

        meshInfo = data.meshInfo.ToArray();
        ComputeHelper.CreateStructuredBuffer(ref modelBuffer, meshInfo);

        // Triangles buffer
        ComputeHelper.CreateStructuredBuffer(ref triangleBuffer, data.triangles);
        rayComputeShader.SetBuffer(kernelRayTrace, "Triangles", triangleBuffer);
        rayComputeShader.SetInt("triangleCount", triangleBuffer.count);

        // Node buffer
        ComputeHelper.CreateStructuredBuffer(ref nodeBuffer, data.nodes);
        rayComputeShader.SetBuffer(kernelRayTrace, "Nodes", nodeBuffer);
    }

    void SetShaderParams()
    {
        rayComputeShader.SetInt("Frame", numAccumulatedFrames);
        rayComputeShader.SetInt("UseSky", useSky ? 1 : 0);

        rayComputeShader.SetInt("MaxBounceCount", maxBounceCount);
        rayComputeShader.SetInt("NumRaysPerPixel", numRaysPerPixel);
        rayComputeShader.SetFloat("DefocusStrength", defocusStrength);
        rayComputeShader.SetFloat("DivergeStrength", divergeStrength);

        rayComputeShader.SetFloat("SunFocus", sunFocus);
        rayComputeShader.SetFloat("SunIntensity", sunIntensity);
        rayComputeShader.SetVector("SunColour", sunColor);
        rayComputeShader.SetVector("dirToSun", sunTransform == null ? Vector3.down : -sunTransform.forward);

        rayComputeShader.SetInt("Frame", numAccumulatedFrames);
        rayComputeShader.SetInt("renderSeed", renderSeed);
        rayComputeShader.SetBool("accumulate", accumulate);

        // also push samplingMode if used
        try { rayComputeShader.SetInt("SamplingMode", samplingMode); } catch { }
    }

    void UpdateCameraParams(Camera cam)
    {
        float planeHeight = focusDistance * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) * 2;
        float planeWidth = planeHeight * cam.aspect;
        // Send data to shader
        rayComputeShader.SetVector("ViewParams", new Vector3(planeWidth, planeHeight, focusDistance));
        rayComputeShader.SetMatrix("CamLocalToWorldMatrix", cam.transform.localToWorldMatrix);
    }

    void UpdateModels()
    {
        for (int i = 0; i < models.Length; i++)
        {
            meshInfo[i].WorldToLocalMatrix = models[i].transform.worldToLocalMatrix;
            meshInfo[i].LocalToWorldMatrix = models[i].transform.localToWorldMatrix;
            meshInfo[i].Material = models[i].material;
        }

        modelBuffer.SetData(meshInfo);
        rayComputeShader.SetBuffer(kernelRayTrace, "ModelInfo", modelBuffer);
        rayComputeShader.SetInt("modelCount", models.Length);
    }

    MeshDataLists CreateAllMeshData(Model[] models)
    {
        MeshDataLists allData = new();
        Dictionary<Mesh, (int nodeOffset, int triOffset)> meshLookup = new();

        foreach (Model model in models)
        {
            // Construct BVH if this is the first time seeing the current mesh (otherwise reuse)
            if (!meshLookup.ContainsKey(model.Mesh))
            {
                meshLookup.Add(model.Mesh, (allData.nodes.Count, allData.triangles.Count));

                BVH bvh = new(model.Mesh.vertices, model.Mesh.triangles, model.Mesh.normals, bvhQuality);
                if (model.logBVHStats) Debug.Log($"BVH Stats: {model.gameObject.name}\n{bvh.stats}");

                allData.triangles.AddRange(bvh.Triangles);
                allData.nodes.AddRange(bvh.Nodes);
            }

            // Create the mesh info
            allData.meshInfo.Add(new MeshInfo()
            {
                NodeOffset = meshLookup[model.Mesh].nodeOffset,
                TriangleOffset = meshLookup[model.Mesh].triOffset,
                WorldToLocalMatrix = model.transform.worldToLocalMatrix,
                Material = model.material
            });
        }

        return allData;
    }

    void OnDestroy()
    {
        if (Application.isPlaying)
        {
            ComputeHelper.Release(triangleBuffer, nodeBuffer, modelBuffer);
            ComputeHelper.Release(accumulatedResult, raytraceFrameTex);
        }

        Seb.Helpers.ComputeHelper.Release(raytraceFrameTex);

        if (_tmpReadbackTex != null) UnityEngine.Object.DestroyImmediate(_tmpReadbackTex);
    }

    class MeshDataLists
    {
        public List<BVH.Triangle> triangles = new();
        public List<BVH.Node> nodes = new();
        public List<MeshInfo> meshInfo = new();
    }

    struct MeshInfo
    {
        public int NodeOffset;
        public int TriangleOffset;
        public Matrix4x4 WorldToLocalMatrix;
        public Matrix4x4 LocalToWorldMatrix;
        public RayTracingMaterial Material;
    }

    // ---------- NEW: per-pixel variance getter (CPU) ----------
    // Returns an array of per-pixel luminance variance values (flattened row-major).
    // Caller owns the array reference (do not modify it in-place if used elsewhere).
    public float[] GetPerPixelVariance()
    {
        if (!computeVarianceCPU)
        {
            Debug.LogWarning("GetPerPixelVariance called but computeVarianceCPU is false.");
            return null;
        }
        if (_sumLuma == null || _sumSqLuma == null) return null;

        int n = Mathf.Max(1, numAccumulatedFrames);
        int count = _sumLuma.Length;
        float[] var = new float[count];

        for (int i = 0; i < count; ++i)
        {
            float mean = _sumLuma[i] / n;
            float meanSq = _sumSqLuma[i] / n;
            float v = meanSq - mean * mean;
            // clamp tiny negative rounding errors
            if (v < 0f && v > -1e-6f) v = 0f;
            var[i] = v;
        }

        return var;
    }
}

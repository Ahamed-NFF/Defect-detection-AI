"use client";

import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import {
  getHealth,
  predict,
  type HealthResponse,
  type PredictResponse,
} from "@/lib/api";
import styles from "./page.module.css";

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e: unknown) =>
        setHealthError(e instanceof Error ? e.message : String(e)),
      );
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function choose(f: File | null) {
    setResult(null);
    setError(null);
    setFile(f);
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    choose(e.target.files?.[0] ?? null);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) choose(f);
  }

  async function runPredict() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predict(file));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const isDefect = result?.label === "defect";
  const confidencePct = result ? Math.round(result.confidence * 100) : 0;

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1>Defect Inspector</h1>
        <p className={styles.subtitle}>
          GAN/Diffusion-augmented manufacturing defect detection
        </p>
      </header>

      {/* backend status banner */}
      {healthError ? (
        <div className={`${styles.banner} ${styles.bannerError}`}>
          Cannot reach the backend API. Start it with{" "}
          <code>uvicorn backend.api.main:app --port 8000</code>. ({healthError})
        </div>
      ) : health ? (
        !health.model_ready ? (
          <div className={`${styles.banner} ${styles.bannerWarn}`}>
            Backend up, but no trained model is loaded — predictions are{" "}
            <strong>not meaningful</strong> yet (dev/untrained weights). Train a
            classifier and set <code>DEFECT_API_CHECKPOINT</code> for real
            results.
          </div>
        ) : (
          <div className={`${styles.banner} ${styles.bannerOk}`}>
            Model ready: {health.model?.backbone} on {health.model?.device}.
          </div>
        )
      ) : (
        <div className={styles.banner}>Checking backend…</div>
      )}

      <section
        className={`${styles.dropzone} ${dragOver ? styles.dropzoneActive : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={onInputChange}
          hidden
        />
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="upload preview"
            className={styles.preview}
          />
        ) : (
          <p>Drag &amp; drop a product image here, or click to browse</p>
        )}
      </section>

      {file && (
        <div className={styles.actions}>
          <span className={styles.filename}>{file.name}</span>
          <button
            className={styles.button}
            onClick={runPredict}
            disabled={loading}
          >
            {loading ? "Inspecting…" : "Inspect"}
          </button>
        </div>
      )}

      {error && (
        <div className={`${styles.banner} ${styles.bannerError}`}>{error}</div>
      )}

      {result && (
        <section className={styles.results}>
          <div className={styles.card}>
            <h2>Result</h2>
            <span
              className={`${styles.badge} ${
                isDefect ? styles.badgeDefect : styles.badgeGood
              }`}
            >
              {result.label.toUpperCase()}
            </span>
            <div className={styles.confidence}>
              <div className={styles.confidenceLabel}>
                <span>confidence</span>
                <span>{confidencePct}%</span>
              </div>
              <div className={styles.bar}>
                <div
                  className={`${styles.barFill} ${
                    isDefect ? styles.barFillDefect : styles.barFillGood
                  }`}
                  style={{ width: `${confidencePct}%` }}
                />
              </div>
              <small className={styles.muted}>
                P(defect) = {result.defect_probability.toFixed(3)}
              </small>
            </div>
            {result.description && (
              <p className={styles.description}>{result.description}</p>
            )}
            {result.model.random_weights && (
              <p className={styles.muted}>
                (untrained model — illustrative only)
              </p>
            )}
          </div>

          <div className={styles.card}>
            <h2>Grad-CAM</h2>
            <p className={styles.muted}>where the model is looking</p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${result.heatmap_png_b64}`}
              alt="Grad-CAM heatmap overlay"
              className={styles.heatmap}
            />
          </div>
        </section>
      )}
    </main>
  );
}

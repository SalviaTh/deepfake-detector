import { useState, useRef, useCallback } from 'react'
import axios from 'axios'
import ReactCrop from 'react-image-crop'
import 'react-image-crop/dist/ReactCrop.css'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '')
const API = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [fileType, setFileType] = useState(null)   // 'image' | 'video'
  const [crop, setCrop] = useState(null)
  const [completedCrop, setCompletedCrop] = useState(null)
  const [trimRange, setTrimRange] = useState([0, 0])
  const [videoDuration, setVideoDuration] = useState(0)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const imgRef = useRef(null)
  const videoRef = useRef(null)

  const onDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer?.files[0] || e.target.files[0]
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
    setFileType(f.type.startsWith('video') ? 'video' : 'image')
  }, [])

  const onVideoLoad = () => {
    const dur = videoRef.current?.duration || 0
    setVideoDuration(dur)
    setTrimRange([0, dur])
  }

  const handleAnalyze = async () => {
    if (!file) return
    console.log("Analyzing content. API URL configured as:", API || "(relative path)")
    setLoading(true)
    setError(null)
    const form = new FormData()

    try {
      if (fileType === 'image' && completedCrop && imgRef.current) {
        // Crop image to canvas then convert to blob
        const canvas = document.createElement('canvas')
        const scaleX = imgRef.current.naturalWidth / imgRef.current.width
        const scaleY = imgRef.current.naturalHeight / imgRef.current.height
        canvas.width = completedCrop.width
        canvas.height = completedCrop.height
        const ctx = canvas.getContext('2d')
        ctx.drawImage(
          imgRef.current,
          completedCrop.x * scaleX, completedCrop.y * scaleY,
          completedCrop.width * scaleX, completedCrop.height * scaleY,
          0, 0, completedCrop.width, completedCrop.height
        )
        const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg'))
        form.append('file', blob, 'crop.jpg')
        const res = await axios.post(`${API}/detect/image`, form)
        setResult({ ...res.data, type: 'image' })
      } else if (fileType === 'image') {
        form.append('file', file)
        const res = await axios.post(`${API}/detect/image`, form)
        setResult({ ...res.data, type: 'image' })
      } else {
        form.append('file', file)
        const params = new URLSearchParams({
          start_sec: trimRange[0],
          end_sec: trimRange[1]
        })
        const res = await axios.post(`${API}/detect/video?${params}`, form)
        setResult({ ...res.data, type: 'video' })
      }
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || "An error occurred during analysis. Make sure the backend is running.")
    } finally {
      setLoading(false)
    }
  }

  const isFake = result?.label === 'FAKE'

  return (
    <div className="app">
      <header>
        <h1>DeepFake Detector</h1>
        <p>Upload an image or video to check authenticity</p>
      </header>

      <main>
        {/* ── Upload zone ── */}
        <div
          className="upload-zone"
          onDrop={onDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => document.getElementById('file-input').click()}
        >
          <input
            id="file-input"
            type="file"
            accept="image/*,video/*"
            style={{ display: 'none' }}
            onChange={onDrop}
          />
          {preview ? (
            <p className="change-hint">✨ Click to change file</p>
          ) : (
            <>
              <div className="upload-icon">📂</div>
              <p>Drag & drop or click to upload</p>
              <p className="sub">Supports JPG, PNG, MP4, MOV</p>
            </>
          )}
        </div>

        {/* ── Image preview with crop ── */}
        {fileType === 'image' && preview && (
          <div className="preview-section">
            <h3>Crop region (optional)</h3>
            <ReactCrop
              crop={crop}
              onChange={c => setCrop(c)}
              onComplete={c => setCompletedCrop(c)}
              aspect={undefined}
            >
              <img ref={imgRef} src={preview} alt="preview"
                   style={{ maxWidth: '100%', maxHeight: 400 }} />
            </ReactCrop>
          </div>
        )}

        {/* ── Video preview with trim ── */}
        {fileType === 'video' && preview && (
          <div className="preview-section">
            <video
              ref={videoRef}
              src={preview}
              controls
              style={{ width: '100%', maxHeight: 360, borderRadius: '12px' }}
              onLoadedMetadata={onVideoLoad}
            />
            <h3>Trim segment</h3>
            <div className="trim-controls">
              <label>Start: {trimRange[0].toFixed(1)}s
                <input type="range" min="0" max={videoDuration}
                  step="0.1" value={trimRange[0]}
                  onChange={e => setTrimRange([+e.target.value, trimRange[1]])}/>
              </label>
              <label>End: {trimRange[1].toFixed(1)}s
                <input type="range" min="0" max={videoDuration}
                  step="0.1" value={trimRange[1]}
                  onChange={e => setTrimRange([trimRange[0], +e.target.value])}/>
              </label>
            </div>
          </div>
        )}

        {/* ── Analyze button ── */}
        {preview && (
          <button className="analyze-btn" onClick={handleAnalyze} disabled={loading}>
            {loading ? '⚡ Analyzing...' : '🔍 Analyze Content'}
          </button>
        )}

        {error && <p className="error">{error}</p>}

        {/* ── Results ── */}
        {result && (
          <div className={`result-card ${isFake ? 'fake' : 'real'}`}>
            <div className="verdict">
              <span className="label">{isFake ? '🚩 FAKE' : '✅ REAL'}</span>
              <span className="confidence">
                {isFake ? 'Manipulation Detected' : 'Content Appears Authentic'}
              </span>
              <h2 style={{ color: isFake ? 'var(--fake)' : 'var(--real)', marginTop: '10px' }}>
                {isFake ? result.confidence : (100 - result.confidence).toFixed(2)}% Fake Probability
              </h2>
            </div>

            {result.type === 'image' && (
              <div className="heatmap-section">
                <div>
                  <p>Analyzed Face</p>
                  <img src={`data:image/jpeg;base64,${result.face_image}`} alt="face"/>
                </div>
                <div>
                  <p>Manipulation Heatmap (Grad-CAM)</p>
                  <img src={`data:image/jpeg;base64,${result.heatmap}`} alt="heatmap"/>
                </div>
              </div>
            )}

            {result.type === 'video' && (
              <div className="timeline">
                <p>Temporal Analysis (Fake Probability %)</p>
                <div className="bars">
                  {result.frame_scores.map(f => (
                    <div key={f.frame} className="bar-wrap"
                         title={`${f.time_sec}s: ${f.prob_fake}%`}>
                      <div className="bar"
                           style={{
                             height: `${f.prob_fake}%`,
                             background: f.prob_fake > 50 ? 'var(--fake)' : 'var(--real)'
                           }}/>
                    </div>
                  ))}
                </div>
                <div className="bar-labels">
                  <span>Start</span><span>End</span>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
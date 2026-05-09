import { useState, useRef } from 'react'
import { uploadPdf } from '../utils/api'
import styles from './PDFUploader.module.css'

function PDFUploader({ onSuccess, onError, disabled }) {
  const [uploading, setUploading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const fileInputRef = useRef(null)

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files)
    if (files.length === 0) return

    files.forEach((file) => {
      if (!file.name.endsWith('.pdf')) {
        onError('Only PDF files are allowed')
        return
      }

      if (file.size > 50 * 1024 * 1024) {
        // 50MB limit
        onError(`File ${file.name} is too large (max 50MB)`)
        return
      }

      uploadFile(file)
    })

    // Reset input
    event.target.value = ''
  }

  const uploadFile = async (file) => {
    setUploading(true)
    try {
      await uploadPdf(file)
      setUploadedFiles((prev) => [...prev, file.name])
      onSuccess(`Successfully uploaded: ${file.name}`)
    } catch (error) {
      onError(error.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    const files = Array.from(e.dataTransfer.files)
    handleFileSelect({ target: { files } })
  }

  return (
    <div className={styles.uploaderContainer}>
      <div className={styles.section}>
        <h2 className={styles.title}>📄 Upload Additional PDFs</h2>

        <div
          className={styles.dropZone}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className={styles.dropZoneContent}>
            <span className={styles.icon}>📤</span>
            <p className={styles.dragText}>
              Drag and drop PDFs here or click to browse
            </p>
            <p className={styles.hint}>Supported: PDF files up to 50MB</p>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileSelect}
          disabled={uploading || disabled}
          style={{ display: 'none' }}
        />

        {uploadedFiles.length > 0 && (
          <div className={styles.filesList}>
            <h3 className={styles.filesTitle}>Uploaded Files:</h3>
            <ul className={styles.files}>
              {uploadedFiles.map((filename, index) => (
                <li key={index} className={styles.fileItem}>
                  ✓ {filename}
                </li>
              ))}
            </ul>
          </div>
        )}

        {uploading && <p className={styles.uploadingText}>Uploading...</p>}
      </div>
    </div>
  )
}

export default PDFUploader

import styles from './LoadingSpinner.module.css'

function LoadingSpinner() {
  return (
    <div className={styles.container}>
      <div className={styles.spinner}></div>
      <span className={styles.text}>Finding the best answer...</span>
    </div>
  )
}

export default LoadingSpinner

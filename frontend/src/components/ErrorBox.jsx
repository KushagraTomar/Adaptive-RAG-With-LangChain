import styles from './ErrorBox.module.css'

function ErrorBox({ message }) {
  return (
    <div className={styles.box}>
      <strong className={styles.title}>⚠️ Error</strong>
      <p className={styles.message}>{message}</p>
    </div>
  )
}

export default ErrorBox

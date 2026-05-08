import styles from './AnswerBox.module.css'

function AnswerBox({ answer }) {
  return (
    <div className={styles.box}>
      <strong className={styles.title}>✓ Answer</strong>
      <p className={styles.content}>{answer.answer}</p>
      <div className={styles.sourceInfo}>
        Source Type: <span className={styles.sourceType}>{answer.source_type || 'hybrid'}</span>
      </div>
    </div>
  )
}

export default AnswerBox

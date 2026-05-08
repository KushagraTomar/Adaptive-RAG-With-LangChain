import { useCallback } from 'react'
import styles from './QuestionInput.module.css'

function QuestionInput({ question, onChangeQuestion, onSubmit, disabled }) {
  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !disabled) {
      onSubmit(e)
    }
  }, [disabled, onSubmit])

  return (
    <form onSubmit={onSubmit} className={styles.form}>
      <input
        type="text"
        value={question}
        onChange={(e) => onChangeQuestion(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Enter your question here..."
        disabled={disabled}
        className={styles.input}
      />
      <button type="submit" disabled={disabled} className={styles.button}>
        {disabled ? 'Loading...' : 'Ask'}
      </button>
    </form>
  )
}

export default QuestionInput

import { useState, useCallback } from 'react'
import QuestionInput from './components/QuestionInput'
import AnswerBox from './components/AnswerBox'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorBox from './components/ErrorBox'
import { askQuestion } from './utils/api'
import styles from './App.module.css'

function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [answer, setAnswer] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault()

    const trimmedQuestion = question.trim()
    if (!trimmedQuestion) {
      setError('Please enter a question')
      return
    }

    setLoading(true)
    setError(null)
    setAnswer(null)

    try {
      const data = await askQuestion(trimmedQuestion)
      setAnswer(data)
      setQuestion('')
    } catch (err) {
      console.error('Error:', err)
      setError(err.message || 'An error occurred while fetching the answer')
    } finally {
      setLoading(false)
    }
  }, [question])

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>🤖 Adaptive RAG Question Answering</h1>

      <QuestionInput
        question={question}
        onChangeQuestion={setQuestion}
        onSubmit={handleSubmit}
        disabled={loading}
      />

      {loading && <LoadingSpinner />}

      {error && <ErrorBox message={error} />}

      {answer && <AnswerBox answer={answer} />}
    </div>
  )
}

export default App

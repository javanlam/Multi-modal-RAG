import { useState } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

interface SourceDocument {
  text?: string;
  metadata?: Record<string, any>;
}

interface QueryResponse {
  question: string;
  answer: string;
  source_documents: SourceDocument[];
  context_images: string[];
  retrieval_metadata: {
    documents_retrieved: number;
    images_retrieved: number;
    enhanced_query?: string;
  };
  generation_metadata: {
    generator_type: string;
  };
}

function App() {
  const [question, setQuestion] = useState('');
  const [useEnhancement, setUseEnhancement] = useState(true);
  const [useVlm, setUseVlm] = useState(true);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          use_enhancement: useEnhancement,
          use_vlm: useVlm,
        }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error (${res.status}): ${errorText}`);
      }

      const data: QueryResponse = await res.json();
      setResponse(data);
    } catch (err: any) {
      setError(err.message || 'Failed to get response from the RAG system.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>📚 Multimodal RAG Demo</h1>
      </header>

      <form onSubmit={handleSubmit} className="query-form">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documents..."
          rows={4}
          required
        />

        <div className="options">
          <label>
            <input
              type="checkbox"
              checked={useEnhancement}
              onChange={(e) => setUseEnhancement(e.target.checked)}
            />
            🧠 Use query enhancement
          </label>
          <label>
            <input
              type="checkbox"
              checked={useVlm}
              onChange={(e) => setUseVlm(e.target.checked)}
            />
            🖼️ Use VLM (Vision Language Model) for image-rich answers
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      {error && <div className="error-message">⚠️ {error}</div>}

      {loading && (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Retrieving and generating answer...</p>
        </div>
      )}

      {response && (
        <div className="results">
          <section className="answer-section">
            <h2>💡 Answer</h2>
            <div className="answer-content">
              {formatAnswer(response.answer)}
            </div>
            <div className="generator-info">
              Generator: {response.generation_metadata.generator_type}
            </div>
            {response.retrieval_metadata.enhanced_query && (
              <div className="enhanced-query">
                ✨ Enhanced query: {response.retrieval_metadata.enhanced_query}
              </div>
            )}
          </section>

          {response.context_images && response.context_images.length > 0 && (
            <section className="images-section">
              <h2>🖼️ Retrieved images ({response.context_images.length})</h2>
              <div className="images-grid">
                {response.context_images.map((url, idx) => (
                  <div key={idx} className="image-card">
                    <img src={url} alt={`Retrieved image ${idx + 1}`} />
                  </div>
                ))}
              </div>
            </section>
          )}

          <section className="sources-section">
            <h2>📄 Source documents ({response.retrieval_metadata.documents_retrieved})</h2>
            <div className="sources-list">
              {response.source_documents.map((doc, idx) => (
                <div key={idx} className="source-card">
                  <div className="source-text">{doc.text || 'No text content'}</div>
                  {doc.metadata && (
                    <div className="source-metadata">
                      {doc.metadata.source_file && <span>📁 {doc.metadata.source_file}</span>}
                      {doc.metadata.page_num && <span>📄 Page {doc.metadata.page_num}</span>}
                      {doc.metadata.has_images && <span>🖼️ Contains images</span>}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

// Simple markdown-like formatting for newlines and inline code
function formatAnswer(text: string): JSX.Element {
  const withLineBreaks = text.split('\n').map((line, i) => (
    <span key={i}>
      {line}
      <br />
    </span>
  ));
  return <div className="answer-text">{withLineBreaks}</div>;
}

export default App;
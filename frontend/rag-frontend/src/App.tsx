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

interface ZeroShotResponse {
  question: string;
  answer: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  model: string;
}

function App() {
  const [question, setQuestion] = useState('');
  const [useEnhancement, setUseEnhancement] = useState(true);
  const [useVlm, setUseVlm] = useState(true);
  const [loading, setLoading] = useState(false);
  const [ragResponse, setRagResponse] = useState<QueryResponse | null>(null);
  const [zeroShotResponse, setZeroShotResponse] = useState<ZeroShotResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setRagResponse(null);
    setZeroShotResponse(null);

    const trimmedQuestion = question.trim();
    const requestBody = {
      question: trimmedQuestion,
      use_enhancement: useEnhancement,
      use_vlm: useVlm,
    };

    try {
      const [ragRes, zeroShotRes] = await Promise.all([
        fetch(`${API_BASE_URL}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        }),
        fetch(`${API_BASE_URL}/query/zeroshot`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: trimmedQuestion }),
        }),
      ]);

      if (!ragRes.ok) {
        const errorText = await ragRes.text();
        throw new Error(`Failed to get responses from the RAG system: (${ragRes.status}): ${errorText}`);
      }
      const ragData: QueryResponse = await ragRes.json();
      setRagResponse(ragData);

      if (!zeroShotRes.ok) {
        const errorText = await zeroShotRes.text();
        throw new Error(`Failed to get responses from the LLM: (${zeroShotRes.status}): ${errorText}`);
      }
      const zeroShotData: ZeroShotResponse = await zeroShotRes.json();
      setZeroShotResponse(zeroShotData);
    } catch (err: any) {
      setError(err.message || 'Failed to get responses from the RAG system.');
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
            🧠 Use query enhancement (RAG only)
          </label>
          <label>
            <input
              type="checkbox"
              checked={useVlm}
              onChange={(e) => setUseVlm(e.target.checked)}
            />
            🖼️ Use VLM (RAG only)
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
          <p>Retrieving and generating answers...</p>
        </div>
      )}

      {ragResponse && zeroShotResponse && (
        <div className="answers-stack">
          {/* Zero‑Shot answer (above) */}
          <div className="answer-block zeroshot-block">
            <h2>⚡ Zero-Shot Answer</h2>
            <div className="answer-content">
              {formatAnswer(zeroShotResponse.answer)}
            </div>
            <div className="model-info">
              Model: {zeroShotResponse.model}
            </div>
            {zeroShotResponse.usage && (
              <div className="usage-info">
                Tokens: {zeroShotResponse.usage.total_tokens} total 
                (prompt: {zeroShotResponse.usage.prompt_tokens}, 
                 completion: {zeroShotResponse.usage.completion_tokens})
              </div>
            )}
            <div className="disclaimer">
              ℹ️ Zero‑shot answers are generated without any document context.
            </div>
          </div>

          {/* RAG‑augmented answer */}
          <div className="answer-block rag-block">
            <h2>🔍 RAG‑augmented Answer</h2>
            <div className="answer-content">
              {formatAnswer(ragResponse.answer)}
            </div>
            <div className="generator-info">
              Generator: {ragResponse.generation_metadata.generator_type}
            </div>
            {ragResponse.retrieval_metadata.enhanced_query && (
              <div className="enhanced-query">
                ✨ Enhanced query: {ragResponse.retrieval_metadata.enhanced_query}
              </div>
            )}

            {ragResponse.context_images && ragResponse.context_images.length > 0 && (
              <div className="images-section">
                <h3>🖼️ Retrieved images ({ragResponse.context_images.length})</h3>
                <div className="images-grid">
                  {ragResponse.context_images.map((url, idx) => (
                    <div key={idx} className="image-card">
                      <img src={url} alt={`Retrieved image ${idx + 1}`} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="sources-section">
              <h3>📄 Source documents ({ragResponse.retrieval_metadata.documents_retrieved})</h3>
              <div className="sources-list">
                {ragResponse.source_documents.map((doc, idx) => (
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
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

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
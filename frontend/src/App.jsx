import { useState } from 'react'
import Header from './Components/Header.jsx'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Header />
      <main>
        <section id="overview" className="section">
          <h1>Use the AI to generate an original composition based on your specifications.</h1>
        </section>

        <section id="output" className="section">
          <h2>Output</h2>
          {/* <Settings onGenerated={(xmlStr, m) => { setXml(xmlStr); setMeta(m) }} /> */}
        </section>

        <section id="prompt" className="section">
          <h2>Select settings</h2>
        </section>
      </main>
      
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
      </div>
    </>
  )
}

export default App

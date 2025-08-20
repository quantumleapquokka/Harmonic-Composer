import { useState } from 'react'
import Header from './Components/Header.jsx'
import './App.css'

function App() {
  const [selectedInstrument, setSelectedInstrument] = useState('')
  const [selectedGenre, setSelectedGenre] = useState('')
  const [selectedEra, setSelectedEra] = useState('')

  const handleChange = (event) => {
    setSelectedInstrument(event.target.value)
  }

  return (
    <>
      <Header />
      <main>
        <section id="overview" className="section">
            <h3>Use the AI to generate an original composition based on your specifications.</h3>
            <h1>Last Composition</h1> 
            {/* <div className="sheet-container">

            </div> */}
        </section>

        <section id="generation" className="section generation-layout">
            <div className="output-side">
            <h2>Output</h2>
            <div className="composition-output">
                <div className="sheet-container">
                        {/* <Settings onGenerated={(xmlStr, m) => { setXml(xmlStr); setMeta(m) }} /> */}
                    </div>

                    <div className="iteration-controls">
                        <button>Regenerate Entirely</button>
                        <button>Keep Rhythm, Change Melody</button>
                        <button>Keep Melody, Change Instrumentation</button>
                        <button>Keep Everything, Just Refine</button>
                    </div>
                </div>
            </div>
            
            
            <div className="settings-side">
                <h2>Select Settings</h2>
                <p>Select your desired specifications for what you want to generate.</p>
                <div className="settings-section">
                    <div className="settings-container">
                        <label>
                            Instrument: <br/>
                            <select value={selectedInstrument} onChange={(e) => setSelectedInstrument(e.target.value)}>
                            <option value="">--Choose Instrument--</option>
                            <option value="piano">Piano</option>
                            <option value="guitar">Guitar</option>
                            </select>
                        </label>
                        <br/>

                        <label>
                            Genre: <br/>
                            <select value={selectedInstrument} onChange={(e) => setSelectedInstrument(e.target.value)}>
                            <option value="">--Choose Genre--</option>
                            <option value="blues">Blues</option>
                            <option value="jazz">Jazz</option>
                            <option value="classical">Classical</option>
                            <option value="country">Country</option>
                            <option value="pop">Pop</option>
                            </select>
                        </label>
                        <br/>

                        <label>
                            Era: <br/>
                            <select value={selectedInstrument} onChange={(e) => setSelectedInstrument(e.target.value)}>
                            <option value="">--Choose Era--</option>
                            <option value="2000s">2000s</option>
                            </select>
                        </label>
                    </div>
                </div>
            </div>

            

            <div className="generate-wrapper">
                <button className="button">Generate Music</button>
            </div>
        </section>

        

        <section id="about" className="section">
          <h2>About / Team</h2>
          <p>CMPM 146 Final Project<br/>AI takes on the role of a collaborative composer, generating, revising, and finalizing music in conjunction with the user.<br/>Brandon Hernandez, Samantha Siew, Shripad Mangavalli, Grishen Hestiyas</p>
        </section>
      </main>
      
    </>
  )
}

export default App
